#!/usr/bin/env python3
"""
Step 3: 结构张量预训练（置信度调度 + 多阶段学习）

加载 datasets/pretrain/ 中结构张量生成的 strand_map + confidence 伪标签，
进行三阶段置信度调度预训练：

  Phase 1 (epoch 0-29): confidence_threshold = 0.8  — 只学核心区域
  Phase 2 (epoch 30-59): confidence_threshold = 0.5  — 加入过渡区域
  Phase 3 (epoch 60-89): confidence_threshold = 0.2  — 全部区域

Usage:
    PYTHONPATH=. pixi run python scripts/pretrain_structure.py \
        --img2strand_backbone hrnet --hrnet_variant hrnet_w32 --hrnet_pretrained \
        --num_epoch 90
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

from lib.options import BaseOptions
from lib.model.img2hairstep.model_factory import create_img2strand_model
from lib.model.img2hairstep.criterion.hairstep_losses import HairStepLoss


# ══════════════════════════════════════════════════════════════
# 置信度调度：根据当前 epoch 返回动态 threshold
# ══════════════════════════════════════════════════════════════
def get_confidence_threshold(epoch: int, total_epochs: int = 90) -> float:
    """
    三阶段调度：
      Phase 1 [0,  30): threshold = 0.8
      Phase 2 [30, 60): threshold = 0.5
      Phase 3 [60, 90]: threshold = 0.2
    """
    if epoch < total_epochs * 0.33:
        return 0.8  # Phase 1: 只学高置信度核心区域
    elif epoch < total_epochs * 0.66:
        return 0.5  # Phase 2: 加入中等置信度
    else:
        return 0.2  # Phase 3: 全部


# ══════════════════════════════════════════════════════════════
# 预训练数据集
# ══════════════════════════════════════════════════════════════
class PretrainDataset(Dataset):
    """
    加载结构张量生成的伪标签数据集。
    目录结构:
      datasets/pretrain/
      ├── img/          (512×512 RGB)
      ├── seg/          (hair mask, 可选)
      ├── strand_map/   ([mask, strand_x, strand_y])
      ├── confidence/   (0-255)
      └── split_train.json
    """

    def __init__(self, data_root, split_file):
        self.data_root = data_root

        with open(split_file, 'r') as f:
            self.items = json.load(f)

        self.img_dir = os.path.join(data_root, 'img')
        self.strand_dir = os.path.join(data_root, 'strand_map')
        self.conf_dir = os.path.join(data_root, 'confidence')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # ── RGB 图片 ──
        rgb_raw = imageio.imread(os.path.join(self.img_dir, item))
        if len(rgb_raw.shape) == 2:
            rgb_raw = np.stack([rgb_raw] * 3, axis=-1)
        rgb_img = rgb_raw[:, :, 0:3].astype(np.float32) / 255.0

        # ── Strand map GT ──
        strand_map_gt = imageio.imread(os.path.join(self.strand_dir, item))
        if strand_map_gt.dtype == np.uint8:
            strand_map_gt = strand_map_gt.astype(np.float32) / 255.0

        # mask = channel 0
        mask = strand_map_gt[:, :, 0]
        mask = (mask > 0.5).astype(np.float32)

        # strand GT = channels 1,2
        strand_gt = strand_map_gt[:, :, 1:3]

        # ── 置信度图 ──
        conf_path = os.path.join(self.conf_dir, item)
        if os.path.exists(conf_path):
            confidence = imageio.imread(conf_path)
            if confidence.dtype == np.uint8:
                confidence = confidence.astype(np.float32) / 255.0
            if confidence.ndim == 3:
                confidence = confidence[:, :, 0]  # 取第一通道
        else:
            confidence = mask  # 无置信度时退化为 mask

        # ── 应用 mask 到 RGB ──
        rgb_img = rgb_img * mask[:, :, None]

        # ── 转 tensor ──
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        strand_gt = torch.from_numpy(strand_gt).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        confidence = torch.from_numpy(confidence).unsqueeze(0).float()

        return rgb_img, strand_gt, mask, confidence


# ══════════════════════════════════════════════════════════════
# 带置信度权重版本的损失计算
# ══════════════════════════════════════════════════════════════
def masked_loss_with_confidence(pred, target, mask, confidence, confidence_threshold):
    """
    在 confidence > threshold 的像素上计算 L1 + Cos loss。
    低于阈值的像素不参与梯度。
    """
    C = pred.shape[1]  # 2
    l1_fn = torch.nn.L1Loss(reduction='none')
    eps = 1e-8

    # ── 有效 mask = hair mask ∩ confidence > threshold ──
    effective_mask = mask * (confidence > confidence_threshold).float()
    total_valid = effective_mask.sum().clamp_min(1.0)

    # ── L1 ──
    l1_per_pixel = l1_fn(pred * effective_mask, target * effective_mask)
    l_l1 = l1_per_pixel.sum() / (total_valid * C).clamp_min(1.0)

    # ── Cos ──
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=eps)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1, eps=eps)
    cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
    cos_loss_map = (1.0 - cos_sim) * effective_mask
    l_cos = cos_loss_map.sum() / total_valid.clamp_min(1.0)

    return l_l1, l_cos, total_valid


# ══════════════════════════════════════════════════════════════
# 预训练主循环
# ══════════════════════════════════════════════════════════════
def get_model_tag(opt):
    backbone = getattr(opt, 'img2strand_backbone', 'unet').lower()
    if backbone == 'hrnet':
        return getattr(opt, 'hrnet_variant', 'hrnet_w18')
    return 'unet'


def pretrain(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = './datasets/pretrain'
    split_file = os.path.join(data_root, 'split_train.json')

    if not os.path.exists(split_file):
        print(f"Error: split file not found at {split_file}")
        print("Run generate_structure_labels.py first.")
        return

    print("Loading pretrain dataset...")
    dataset = PretrainDataset(data_root, split_file)
    print(f"  Samples: {len(dataset)}")

    loader_kwargs = {
        'batch_size': opt.batch_size,
        'shuffle': True,
        'num_workers': opt.num_threads,
        'pin_memory': opt.pin_memory,
        'drop_last': True,
    }
    if opt.num_threads > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    loader = DataLoader(dataset, **loader_kwargs)

    # ── 模型 ──
    model = create_img2strand_model(opt).to(device)

    model_tag = get_model_tag(opt)
    save_dir = os.path.join('./checkpoints/pretrain', model_tag)
    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    print(f"Loss: L1 + Cos (confidence-gated)")
    print(f"Confidence schedule: epoch 0-29→0.8, 30-59→0.5, 60-89→0.2")
    print(f"Starting pretraining for {opt.num_epoch} epochs...")

    for epoch in range(opt.num_epoch):
        model.train()
        conf_thresh = get_confidence_threshold(epoch, opt.num_epoch)

        running_l1, running_cos, running_total = 0.0, 0.0, 0.0
        safe_pixels_ratio = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{opt.num_epoch - 1} (th={conf_thresh:.1f})")
        for rgb_img, strand_gt, mask, confidence in pbar:
            rgb_img = rgb_img.to(device)
            strand_gt = strand_gt.to(device)
            mask = mask.to(device)
            confidence = confidence.to(device)

            optimizer.zero_grad()

            out = model(rgb_img)
            pred, _ = out if isinstance(out, tuple) else (out, None)

            l1, cos, valid_count = masked_loss_with_confidence(
                pred, strand_gt, mask, confidence, conf_thresh
            )
            w_l1 = getattr(opt, 'w_l1', 1.0)
            w_cos = getattr(opt, 'w_cos', 1.0)
            loss = w_l1 * l1 + w_cos * cos
            loss.backward()
            optimizer.step()

            running_l1 += l1.item()
            running_cos += cos.item()
            running_total += loss.item()
            safe_pixels_ratio += valid_count.item() / max(1, mask.sum().item())
            n += 1

            pbar.set_postfix({
                'L1': f'{l1.item():.3f}',
                'cos': f'{cos.item():.3f}',
                'tot': f'{loss.item():.3f}',
            })

        n = max(n, 1)
        print(f"Epoch {epoch} Avg | threshold={conf_thresh:.1f} | "
              f"total={running_total / n:.4f}  l1={running_l1 / n:.4f}  "
              f"cos={running_cos / n:.4f}  "
              f"active_pixels={safe_pixels_ratio / n * 100:.1f}%")

        # ── 保存 checkpoint ──
        if epoch % opt.freq_save == 0 or epoch == opt.num_epoch - 1:
            save_path = os.path.join(save_dir, f'img2strand_{model_tag}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    opt = BaseOptions().parse()
    if opt.batch_size == 1:
        opt.batch_size = 8
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    pretrain(opt)