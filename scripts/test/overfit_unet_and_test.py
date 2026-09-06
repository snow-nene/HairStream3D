#!/usr/bin/env python3
"""
UNet 单图过拟合 → 196 张测试集泛化性能评估

步骤:
  1. 从训练集取 1 张图，UNet 过拟合训练 (L1+Cos loss, ~500 epochs)
  2. 在 196 张测试集上推理
  3. 输出可视化面板: [原图 | gt_strandmap | pred_strandmap]

strandmap 可视化: HSV 方向场色轮编码
  - Hue = atan2(y, x)   → 方向映射为色相
  - Value = magnitude    → 强度

Usage:
    PYTHONPATH=. pixi run python scripts/test/overfit_unet_and_test.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

from lib.model.img2hairstep.UNet import Model as UNet
from scripts.train.train_hisa import HiSaDataset


# ═══════════════════════════════════════════════════════════════
# Strandmap 可视化 (参考 ReChannel eval.py)
#   5 列拼图: [原图 | GT strandmap RGB | Pred strandmap RGB | GT overlay 红线 | Pred overlay 绿线]
# ═══════════════════════════════════════════════════════════════
from PIL import Image, ImageDraw


def _strandmap_2ch_to_rgb(strand_2ch, mask):
    """模型输出 2ch [0,1] → strandmap RGB (R=mask, G=dy_enc, B=dx_enc)"""
    m = mask[:, :, 0]
    r = (m * 255).astype(np.uint8)
    g = np.clip(strand_2ch[:, :, 0] * m * 255, 0, 255).astype(np.uint8)
    b = np.clip(strand_2ch[:, :, 1] * m * 255, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _decode_dx_dy_from_sm_file(sm_rgb):
    """从 strand_map PNG 文件解码 dx, dy"""
    g = sm_rgb[:, :, 1].astype(np.float32)
    b = sm_rgb[:, :, 2].astype(np.float32)
    dy = (g / 255.0) * 2.0 - 1.0
    dx = -((b / 255.0) * 2.0 - 1.0)
    return dx, dy


def _decode_dx_dy_from_pred(pred_2ch):
    """从模型输出 2ch [0,1] 解码 dx, dy"""
    dy = pred_2ch[:, :, 0].astype(np.float32) * 2.0 - 1.0
    dx = 1.0 - pred_2ch[:, :, 1].astype(np.float32) * 2.0
    return dx, dy


def _draw_strands(rgb, dx, dy, mask, step=6, line_len=7, color=(220, 40, 40), width=1):
    """在 RGB 图上画方向线段 (PIL)。mask 是 (H,W) bool。"""
    mag = np.sqrt(dx * dx + dy * dy) + 1e-8
    H, W = mask.shape
    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            if not mask[y, x] or mag[y, x] < 0.05:
                continue
            lx = dx[y, x] / mag[y, x] * line_len
            ly = dy[y, x] / mag[y, x] * line_len
            draw.line([(x - lx, y - ly), (x + lx, y + ly)],
                      fill=color, width=width)
    return np.array(out)


def _overlay_from_sm_file(rgb, sm_rgb, mask_bool, step=6, line_len=7):
    """GT strand_map PNG → 解码 → 红线叠加"""
    dx, dy = _decode_dx_dy_from_sm_file(sm_rgb)
    return _draw_strands(rgb, dx, dy, mask_bool, step, line_len,
                         color=(220, 40, 40), width=1)


def _overlay_from_pred(rgb, pred_2ch, mask_bool, step=6, line_len=7):
    """模型输出 2ch → 解码 → 绿线叠加"""
    dx, dy = _decode_dx_dy_from_pred(pred_2ch)
    return _draw_strands(rgb, dx, dy, mask_bool, step, line_len,
                         color=(60, 180, 60), width=1)


# ═══════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════
def compute_loss(pred, target, mask):
    """L1 + Cosine similarity loss"""
    eps = 1e-8
    C = pred.shape[1]

    # L1 (masked MAE)
    l1_per_pixel = torch.abs((pred - target) * mask)
    l1 = l1_per_pixel.sum() / (mask.sum() * C).clamp_min(1.0)

    # Cosine loss
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=eps)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1, eps=eps)
    cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
    cos = ((1.0 - cos_sim) * mask).sum() / mask.sum().clamp_min(1.0)

    return l1, cos


# ═══════════════════════════════════════════════════════════════
# 过拟合训练: 单张图片
# ═══════════════════════════════════════════════════════════════
def overfit_single_image(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = './datasets/HiSa_HiDa'
    train_split = os.path.join(data_root, 'split_train.json')

    with open(train_split, 'r') as f:
        train_items = json.load(f)

    # 取第一张图
    sample_name = train_items[0]
    print(f"Overfitting on single sample: {sample_name}")

    # 用 HiSaDataset 只取这一张
    dataset = HiSaDataset(data_root, train_split, augment=False)
    # 找到这个 sample 的 index
    sample_idx = train_items.index(sample_name)
    single_dataset = Subset(dataset, [sample_idx])
    loader = DataLoader(single_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 取出这张图的数据
    rgb_img, strand_gt, mask = next(iter(loader))
    rgb_img = rgb_img.to(device)
    strand_gt = strand_gt.to(device)
    mask = mask.to(device)

    print(f"  Image shape: {rgb_img.shape}")
    print(f"  Strand GT shape: {strand_gt.shape}")
    print(f"  Hair pixels: {mask.sum().item():.0f} / {mask.numel()} ({100*mask.sum().item()/mask.numel():.1f}%)")

    # 模型
    model = UNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet params: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    save_dir = os.path.join('./checkpoints/overfit_unet')
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = opt.overfit_epochs
    print(f"\nOverfitting for {num_epochs} epochs...")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(rgb_img)
        # UNet returns single tensor, not tuple
        pred = out if not isinstance(out, tuple) else out[0]

        l1, cos = compute_loss(pred, strand_gt, mask)
        loss = opt.w_l1 * l1 + opt.w_cos * cos
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_path = os.path.join(save_dir, 'img2strand_unet_overfit_best.pth')
            torch.save(model.state_dict(), best_path)

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            # 计算 mask 内平均角度误差
            with torch.no_grad():
                pred_n = torch.nn.functional.normalize(pred, p=2, dim=1)
                gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1)
                cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
                angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
                mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
                mean_angle_deg = np.rad2deg(mean_angle)

            print(f"  Epoch {epoch:4d}/{num_epochs} | "
                  f"loss={loss_val:.6f}  l1={l1.item():.6f}  cos={cos.item():.6f}  "
                  f"angle_err={mean_angle_deg:.2f}°")

    # 最终保存
    final_path = os.path.join(save_dir, 'img2strand_unet_overfit_final.pth')
    torch.save(model.state_dict(), final_path)

    print(f"\nBest loss: {best_loss:.6f} → {best_path}")
    print(f"Final checkpoint: {final_path}")

    # 输出收敛分析
    if len(loss_history) >= 100:
        first_100 = np.mean(loss_history[:10])
        last_100 = np.mean(loss_history[-10:])
        print(f"Loss reduction: {first_100:.6f} → {last_100:.6f} ({100*(1-last_100/first_100):.1f}% reduction)")

    # 打印最终单图角度误差
    with torch.no_grad():
        model.eval()
        pred = model(rgb_img)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_n = torch.nn.functional.normalize(pred, p=2, dim=1)
        gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1)
        cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
        angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
        mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
        print(f"\nSingle image final angle error: {np.rad2deg(mean_angle):.4f}°")

    return final_path


# ═══════════════════════════════════════════════════════════════
# 测试: 196 张测试集 + 可视化输出
# ═══════════════════════════════════════════════════════════════
def test_and_visualize(ckpt_path, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = './datasets/HiSa_HiDa'
    test_split = os.path.join(data_root, 'split_test.json')

    with open(test_split, 'r') as f:
        test_items = json.load(f)
    print(f"\nTest set: {len(test_items)} images")

    # 加载数据集（用于拿到处理好的 tensor）
    dataset = HiSaDataset(data_root, test_split, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 加载过拟合模型
    model = UNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    vis_dir = os.path.join('./results/overfit_unet_test')
    os.makedirs(vis_dir, exist_ok=True)

    img_dir = os.path.join(data_root, 'img')
    strand_dir = os.path.join(data_root, 'strand_map')

    all_l1 = []
    all_cos = []
    all_angle_deg = []
    panel_rows = []  # for stitched panel

    N_PANEL_ROWS = 16  # stitch first 16 images into one big panel

    print(f"Saving visualizations to {vis_dir}/ ...")

    for idx, (rgb_img, strand_gt, mask) in enumerate(tqdm(loader, desc="Testing")):
        rgb_img = rgb_img.to(device)
        strand_gt = strand_gt.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            out = model(rgb_img)
            pred = out if not isinstance(out, tuple) else out[0]

        # 计算指标
        l1, cos = compute_loss(pred, strand_gt, mask)
        all_l1.append(l1.item())
        all_cos.append(cos.item())

        # 角度误差
        pred_n = torch.nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
        gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1, eps=1e-8)
        cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
        angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
        mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
        all_angle_deg.append(np.rad2deg(mean_angle))

        sample_name = test_items[idx]

        # ── 原图 ──
        raw_rgb = imageio.imread(os.path.join(img_dir, sample_name))
        if raw_rgb.ndim == 3 and raw_rgb.shape[2] >= 3:
            raw_rgb = raw_rgb[:, :, :3]
        elif raw_rgb.ndim == 2:
            raw_rgb = np.stack([raw_rgb] * 3, axis=-1)

        mask_np = mask.cpu().squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 1)
        mask_bool = mask_np[:, :, 0] > 0.5
        pred_np = pred.cpu().squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 2)

        # ── GT strandmap RGB (原始文件 + mask 抠图) ──
        gt_sm_file = imageio.imread(os.path.join(strand_dir, sample_name))
        if gt_sm_file.ndim == 3 and gt_sm_file.shape[2] >= 3:
            gt_sm_file = gt_sm_file[:, :, :3]
        gt_rgb = (gt_sm_file.astype(np.float32) * mask_np).astype(np.uint8)

        # ── Pred strandmap RGB ──
        pd_rgb = _strandmap_2ch_to_rgb(pred_np, mask_np)

        # ── GT overlay (红线) ──
        gt_overlay = _overlay_from_sm_file(raw_rgb, gt_sm_file, mask_bool)

        # ── Pred overlay (绿线) ──
        pd_overlay = _overlay_from_pred(raw_rgb, pred_np, mask_bool)

        # ── 5 列面板 ──
        panel = np.concatenate([raw_rgb, gt_rgb, pd_rgb, gt_overlay, pd_overlay], axis=1)

        out_path = os.path.join(vis_dir, sample_name)
        Image.fromarray(panel).save(out_path)

        if idx < N_PANEL_ROWS:
            panel_rows.append(panel)

    # ── 保存 stitched panel ──
    if panel_rows:
        big_panel = np.concatenate(panel_rows, axis=0)  # vertical stack
        Image.fromarray(big_panel).save(os.path.join(vis_dir, "panel.png"))
        print(f"Stitched panel ({len(panel_rows)} rows) saved: {vis_dir}/panel.png")

    # ── 统计 ──
    mean_l1 = np.mean(all_l1)
    mean_cos = np.mean(all_cos)
    mean_angle = np.mean(all_angle_deg)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS (196 images, UNet overfitted on 1 image)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    print(f"  Mean L1 (masked MAE):     {mean_l1:.6f}")
    print(f"  Mean Cos loss:             {mean_cos:.6f}")
    print(f"  Mean angle error:          {mean_angle:.4f}°")
    print(f"  Median angle error:        {np.median(all_angle_deg):.4f}°")
    print(f"  Std angle error:           {np.std(all_angle_deg):.4f}°")
    print(f"  Best angle error:          {np.min(all_angle_deg):.4f}°")
    print(f"  Worst angle error:         {np.max(all_angle_deg):.4f}°")
    print(f"{'='*60}")
    print(f"  Visualizations saved: {vis_dir}/")

    # 保存 metrics JSON
    metrics = {
        'ckpt': ckpt_path,
        'mean_l1': float(mean_l1),
        'mean_cos': float(mean_cos),
        'mean_angle_deg': float(mean_angle),
        'median_angle_deg': float(np.median(all_angle_deg)),
        'std_angle_deg': float(np.std(all_angle_deg)),
        'best_angle_deg': float(np.min(all_angle_deg)),
        'worst_angle_deg': float(np.max(all_angle_deg)),
        'per_sample_angle_deg': [float(x) for x in all_angle_deg],
    }
    metrics_path = os.path.join(vis_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return mean_l1, mean_cos, mean_angle


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='UNet overfit on 1 image → test on 196')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for overfitting')
    parser.add_argument('--overfit_epochs', type=int, default=500, help='Epochs to overfit on single image')
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_cos', type=float, default=1.0)
    parser.add_argument('--skip_train', action='store_true', help='Skip overfitting, use existing checkpoint')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/overfit_unet/img2strand_unet_overfit_best.pth',
                        help='Checkpoint path (for --skip_train or to override)')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.skip_train:
        ckpt_path = args.ckpt
        if not os.path.exists(ckpt_path):
            print(f"Error: checkpoint not found at {ckpt_path}")
            sys.exit(1)
        print(f"Skipping training, using: {ckpt_path}")
    else:
        ckpt_path = overfit_single_image(args)

    test_and_visualize(ckpt_path, args)


if __name__ == "__main__":
    main()
