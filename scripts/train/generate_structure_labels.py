#!/usr/bin/env python3
"""
Step 2: 批量生成 0-180° 方向场 + 置信度图（结构张量）

对 datasets/pretrain/img/ 中每张图：
  1. 读取 SAM mask
  2. 调用 standard_structure_tensor_pipeline 获取 theta (0-π) 和 coherence
  3. 输出 strand_map (3通道: [mask, strand_x, strand_y]) 和 confidence 图

Usage:
    PYTHONPATH=. pixi run python scripts/train/generate_structure_labels.py \
        --input_dir datasets/pretrain \
        --output_dir datasets/pretrain
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── 直接调用 standard_pipeline.py 的 actual API ──
from standard_pipeline import standard_structure_tensor_pipeline


def process_single_image(img_path, seg_path, sigma_integration,
                         out_strand_dir, out_conf_dir):
    """
    单张图片处理：
      1. 读取全图 + mask
      2. 调用 standard_structure_tensor_pipeline(rgb, mask, sigma) → theta, coherence
      3. theta → (cosθ, sinθ) → strand_map
      4. coherence → confidence
    """
    # ── 读取全图 (RGB, [0,1] 归一化) ──
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception:
        return False
    # 确保 512×512
    img_pil = img_pil.resize((512, 512), Image.Resampling.LANCZOS)
    rgb = np.asarray(img_pil, dtype=np.float32) / 255.0

    # ── 读取 mask ──
    if os.path.exists(seg_path):
        mask_pil = Image.open(seg_path).convert("L")
        mask_pil = mask_pil.resize((512, 512), Image.Resampling.NEAREST)
        mask = np.array(mask_pil) > 127
    else:
        mask = np.ones((512, 512), dtype=bool)

    # ── 结构张量 ──
    try:
        theta, coherence = standard_structure_tensor_pipeline(
            rgb, mask, sigma_integration=sigma_integration
        )
    except Exception:
        return False

    # ── mask 区域置零 ──
    coherence = np.where(mask, coherence, 0.0).astype(np.float32)
    final_mask = mask.astype(np.float32)

    # ── theta [0, π) → (cosθ, sinθ) 方向向量 ──
    # 注意: theta 表示头发切线方向，cos(θ) 是 x 分量，sin(θ) 是 y 分量
    field_x = np.cos(theta).astype(np.float32)
    field_y = np.sin(theta).astype(np.float32)
    # mask 外用零
    field_x = np.where(mask, field_x, 0.0)
    field_y = np.where(mask, field_y, 0.0)

    # ── 构造 strand_map (3通道) ──
    # R = mask (0 或 255)
    # G = (field_x + 1) / 2 * 255  → cosθ → [0, 255]
    # B = (field_y + 1) / 2 * 255  → sinθ → [0, 255]
    strand_map = np.zeros((512, 512, 3), dtype=np.uint8)
    strand_map[:, :, 0] = (final_mask * 255).astype(np.uint8)

    masked_x = field_x * final_mask
    masked_y = field_y * final_mask
    strand_map[:, :, 1] = np.clip((masked_x + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    strand_map[:, :, 2] = np.clip((masked_y + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    # ── 置信度 = coherence × final_mask ──
    confidence = coherence * final_mask

    # ── 保存 ──
    basename = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_strand_dir, f'{basename}.png'), strand_map)

    conf_img = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_conf_dir, f'{basename}.png'), conf_img)

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate structure tensor labels')
    parser.add_argument('--input_dir', default='datasets/pretrain',
                        help='Directory containing img/ and seg/ subdirectories')
    parser.add_argument('--output_dir', default='datasets/pretrain',
                        help='Output directory for strand_map/ confidence/')
    parser.add_argument('--sigma_integration', type=float, default=4.0,
                        help='Gaussian integration sigma for structure tensor')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit to N images (0 = all). Use for smoke test.')
    args = parser.parse_args()

    img_dir = os.path.join(args.input_dir, 'img')
    seg_dir = os.path.join(args.input_dir, 'seg')
    strand_dir = os.path.join(args.output_dir, 'strand_map')
    conf_dir = os.path.join(args.output_dir, 'confidence')

    os.makedirs(strand_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    # ── 收集图片列表 ──
    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.max_samples > 0:
        img_files = img_files[:args.max_samples]

    print(f"Found {len(img_files)} images in {img_dir}")
    print(f"sigma_integration={args.sigma_integration}")

    success = 0
    valid_items = []

    for fname in tqdm(img_files, desc="Structure tensor"):
        img_path = os.path.join(img_dir, fname)
        seg_path = os.path.join(seg_dir, fname)

        ok = process_single_image(
            img_path, seg_path,
            args.sigma_integration,
            strand_dir, conf_dir,
        )
        if ok:
            success += 1
            valid_items.append(fname)

    # ── 写入 split 文件 ──
    split_path = os.path.join(args.output_dir, 'split_train.json')
    with open(split_path, 'w') as f:
        json.dump(valid_items, f, indent=2)

    print(f"\nDone: {success}/{len(img_files)} succeeded")
    print(f"Split file: {split_path} ({len(valid_items)} entries)")


if __name__ == '__main__':
    main()