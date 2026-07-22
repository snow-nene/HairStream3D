#!/usr/bin/env python3
"""
HRNet-W32 单图过拟合 → 196 张测试集泛化性能评估

步骤:
  1. 从训练集取 1 张图，HRNet-W32 过拟合训练 (L1+Cos loss)
  2. 在 196 张测试集上推理
  3. 输出 L1/Cos/Angle error 指标

Usage:
    PYTHONPATH=. pixi run python scripts/overfit_hrnet_w32_and_test.py
    PYTHONPATH=. pixi run python scripts/overfit_hrnet_w32_and_test.py --overfit_epochs 2000
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from lib.model.img2hairstep.model_factory import create_img2strand_model
from lib.options import BaseOptions
from scripts.train_hisa import HiSaDataset


# ═══════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════
def compute_loss(pred, target, mask):
    """L1 + Cosine similarity loss"""
    eps = 1e-8
    C = pred.shape[1]

    # L1 (masked MAE): (1 / (C * N_mask)) * Σ|pred - gt|
    l1_per_pixel = torch.abs((pred - target) * mask)
    l1 = l1_per_pixel.sum() / (mask.sum() * C).clamp_min(1.0)

    # Cosine loss
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=eps)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1, eps=eps)
    cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
    cos = ((1.0 - cos_sim) * mask).sum() / mask.sum().clamp_min(1.0)

    return l1, cos


# ═══════════════════════════════════════════════════════════════
# 过拟合训练: HRNet-W32 单张图片
# ═══════════════════════════════════════════════════════════════
def overfit_single_image(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = './datasets/HiSa_HiDa'
    train_split = os.path.join(data_root, 'split_train.json')

    with open(train_split, 'r') as f:
        train_items = json.load(f)

    sample_name = train_items[0]
    print(f"Overfitting on single sample: {sample_name}")

    dataset = HiSaDataset(data_root, train_split, augment=False)
    sample_idx = train_items.index(sample_name)
    single_dataset = Subset(dataset, [sample_idx])
    loader = DataLoader(single_dataset, batch_size=1, shuffle=False, num_workers=0)

    rgb_img, strand_gt, mask = next(iter(loader))
    rgb_img = rgb_img.to(device)
    strand_gt = strand_gt.to(device)
    mask = mask.to(device)

    print(f"  Image shape: {rgb_img.shape}")
    print(f"  Strand GT shape: {strand_gt.shape}")
    print(f"  Hair pixels: {mask.sum().item():.0f} / {mask.numel()} ({100*mask.sum().item()/mask.numel():.1f}%)")

    # HRNet-W32 模型（不消费 sys.argv）
    import copy
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    opt_base = BaseOptions().parse()
    sys.argv = old_argv
    opt_base.img2strand_backbone = 'hrnet'
    opt_base.hrnet_variant = 'hrnet_w32'
    opt_base.multi_scale_supervision = True

    model = create_img2strand_model(opt_base).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  HRNet-W32 params: {total_params:,} (trainable: {trainable_params:,})")

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    save_dir = './checkpoints/overfit_hrnet_w32'
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = opt.overfit_epochs
    print(f"\nOverfitting for {num_epochs} epochs...")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(rgb_img)
        # HRNet returns (pred_main, aux_preds)
        pred = out[0] if isinstance(out, tuple) else out

        l1, cos = compute_loss(pred, strand_gt, mask)
        loss = opt.w_l1 * l1 + opt.w_cos * cos
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_path = os.path.join(save_dir, 'img2strand_hrnet_w32_overfit_best.pth')
            torch.save(model.state_dict(), best_path)

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                pred_n = torch.nn.functional.normalize(pred, p=2, dim=1)
                gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1)
                cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
                angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
                mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
                mean_angle_deg = np.rad2deg(mean_angle)

            print(f"  Epoch {epoch:5d}/{num_epochs} | "
                  f"loss={loss_val:.6f}  l1={l1.item():.6f}  cos={cos.item():.6f}  "
                  f"angle_err={mean_angle_deg:.2f}°")

    final_path = os.path.join(save_dir, 'img2strand_hrnet_w32_overfit_final.pth')
    torch.save(model.state_dict(), final_path)

    print(f"\nBest loss: {best_loss:.6f} → {best_path}")
    print(f"Final checkpoint: {final_path}")

    if len(loss_history) >= 100:
        first_10 = np.mean(loss_history[:10])
        last_10 = np.mean(loss_history[-10:])
        print(f"Loss reduction: {first_10:.6f} → {last_10:.6f} ({100*(1-last_10/max(first_10,1e-8)):.1f}% reduction)")

    # 最终单图误差
    with torch.no_grad():
        model.eval()
        out = model(rgb_img)
        pred = out[0] if isinstance(out, tuple) else out
        pred_n = torch.nn.functional.normalize(pred, p=2, dim=1)
        gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1)
        cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
        angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
        mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
        print(f"\nSingle image final angle error: {np.rad2deg(mean_angle):.4f}°")

    return final_path, sample_name


# ═══════════════════════════════════════════════════════════════
# 测试: 196 张测试集
# ═══════════════════════════════════════════════════════════════
def test_and_report(ckpt_path, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = './datasets/HiSa_HiDa'
    test_split = os.path.join(data_root, 'split_test.json')

    with open(test_split, 'r') as f:
        test_items = json.load(f)
    print(f"\nTest set: {len(test_items)} images")

    dataset = HiSaDataset(data_root, test_split, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    import copy
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    opt_base = BaseOptions().parse()
    sys.argv = old_argv
    opt_base.img2strand_backbone = 'hrnet'
    opt_base.hrnet_variant = 'hrnet_w32'
    opt_base.multi_scale_supervision = True

    model = create_img2strand_model(opt_base).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_l1 = []
    all_cos = []
    all_angle_deg = []

    for rgb_img, strand_gt, mask in tqdm(loader, desc="Testing"):
        rgb_img = rgb_img.to(device)
        strand_gt = strand_gt.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            out = model(rgb_img)
            pred = out[0] if isinstance(out, tuple) else out

        l1, cos = compute_loss(pred, strand_gt, mask)
        all_l1.append(l1.item())
        all_cos.append(cos.item())

        pred_n = torch.nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
        gt_n = torch.nn.functional.normalize(strand_gt, p=2, dim=1, eps=1e-8)
        cos_sim = (pred_n * gt_n).sum(dim=1, keepdim=True)
        angle_err = torch.acos(cos_sim.clamp(-1+1e-7, 1-1e-7))
        mean_angle = ((angle_err * mask).sum() / mask.sum().clamp_min(1.0)).item()
        all_angle_deg.append(np.rad2deg(mean_angle))

    mean_l1 = np.mean(all_l1)
    mean_cos = np.mean(all_cos)
    mean_angle = np.mean(all_angle_deg)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS (196 images, HRNet-W32 overfitted on 1 image)")
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

    metrics_dir = './results/overfit_hrnet_w32_test'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return mean_l1, mean_cos, mean_angle


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='HRNet-W32 overfit on 1 image → test on 196')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--overfit_epochs', type=int, default=500, help='Epochs to overfit on single image')
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_cos', type=float, default=1.0)
    parser.add_argument('--skip_train', action='store_true', help='Skip overfitting, use existing checkpoint')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/overfit_hrnet_w32/img2strand_hrnet_w32_overfit_best.pth',
                        help='Checkpoint path (for --skip_train)')
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
        ckpt_path, sample_name = overfit_single_image(args)

    test_and_report(ckpt_path, args)


if __name__ == "__main__":
    main()
