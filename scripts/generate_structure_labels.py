#!/usr/bin/env python3
"""
Step 2: 批量生成 0-180° 方向场 + 置信度图（结构张量）

对 datasets/pretrain/img/ 中每张图：
  1. 调用 extract_structure_field 获取 field_x, field_y, coherence, hair_mask
  2. 融合 SAM mask 和结构张量 mask
  3. 输出 strand_map (3通道: [mask, strand_x, strand_y]) 和 confidence 图

Usage:
    PYTHONPATH=. pixi run python scripts/generate_structure_labels.py \
        --input_dir datasets/pretrain \
        --output_dir datasets/pretrain
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from scripts.standard_pipeline import (
    extract_structure_field,
    read_image,
    normalize_image,
    to_gray,
    _calc_structure_tensor,
    compute_coherence,
)


def _extract_dense_structure_tensor(img_rgb, sigma_inner, sigma_outer):
    """
    提取全图密集的结构张量方向场（不追踪流线，直接用张量方向）。
    返回: field_x, field_y (H, W), coherence (H, W), hair_mask (H, W)
    """
    gray = to_gray(img_rgb)
    gray_norm = normalize_image(gray)

    lambda1, lambda2, vx, vy = _calc_structure_tensor(
        gray_norm, sigma_inner=sigma_inner, sigma_outer=sigma_outer
    )
    coherence = compute_coherence(lambda1, lambda2)

    # 全图 dense: field = (vx, vy) 就是方向向量
    # 归一化已经在 _calc_structure_tensor 中完成

    # 基于 coherence + brightness 的 hair mask
    hair_mask = (coherence > 0.1) & (gray_norm > 0.05)

    return vx.astype(np.float32), vy.astype(np.float32), coherence.astype(np.float32), hair_mask


def process_single_image(img_path, seg_path, sigma_inner, sigma_outer,
                         out_strand_dir, out_conf_dir, out_coherence_dir):
    """处理单张图片，输出 strand_map 和 confidence"""
    img = read_image(img_path)
    if img is None:
        return False

    H, W = img.shape[:2]

    # ── 读取 SAM mask ──
    if os.path.exists(seg_path):
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        seg_mask = (seg_mask > 128).astype(np.float32)
    else:
        seg_mask = np.ones((H, W), dtype=np.float32)

    # ── 密集结构张量 ──
    try:
        vx, vy, coherence, struct_mask = _extract_dense_structure_tensor(
            img, sigma_inner, sigma_outer
        )
    except Exception:
        return False

    # ── 融合 mask：SAM ∩ 结构张量 mask ──
    final_mask = seg_mask * struct_mask.astype(np.float32)

    # ── 置信度 = coherence × final_mask ──
    confidence = coherence * final_mask

    # ── 构造 strand_map (3通道) ──
    # R = mask (0 或 1)
    # G = (field_x + 1) / 2 * 255  → cos θ → [0, 255]
    # B = (field_y + 1) / 2 * 255  → sin θ → [0, 255]
    strand_map = np.zeros((H, W, 3), dtype=np.uint8)
    strand_map[:, :, 0] = (final_mask * 255).astype(np.uint8)

    # 只对 mask 区域编码
    masked_x = vx * final_mask
    masked_y = vy * final_mask
    strand_map[:, :, 1] = np.clip((masked_x + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    strand_map[:, :, 2] = np.clip((masked_y + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    # ── 保存 ──
    basename = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_strand_dir, f'{basename}.png'), strand_map)

    # confidence: 0-1 → 0-255
    conf_img = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_conf_dir, f'{basename}.png'), conf_img)

    # coherence（仅用于可视化/调试）
    coh_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_coherence_dir, f'{basename}.png'), coh_img)

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate structure tensor labels')
    parser.add_argument('--input_dir', default='datasets/pretrain',
                        help='Directory containing img/ and seg/ subdirectories')
    parser.add_argument('--output_dir', default='datasets/pretrain',
                        help='Output directory for strand_map/ confidence/ coherence/')
    parser.add_argument('--sigma_inner', type=float, default=3.0,
                        help='Inner Gaussian sigma for structure tensor')
    parser.add_argument('--sigma_outer', type=float, default=9.0,
                        help='Outer Gaussian sigma for structure tensor')
    args = parser.parse_args()

    img_dir = os.path.join(args.input_dir, 'img')
    seg_dir = os.path.join(args.input_dir, 'seg')
    strand_dir = os.path.join(args.output_dir, 'strand_map')
    conf_dir = os.path.join(args.output_dir, 'confidence')
    coh_dir = os.path.join(args.output_dir, 'coherence')

    os.makedirs(strand_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(coh_dir, exist_ok=True)

    # ── 收集图片列表 ──
    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    print(f"Found {len(img_files)} images in {img_dir}")
    print(f"sigma_inner={args.sigma_inner}, sigma_outer={args.sigma_outer}")

    success = 0
    valid_items = []

    for fname in tqdm(img_files, desc="Structure tensor"):
        img_path = os.path.join(img_dir, fname)
        seg_path = os.path.join(seg_dir, fname)

        ok = process_single_image(
            img_path, seg_path,
            args.sigma_inner, args.sigma_outer,
            strand_dir, conf_dir, coh_dir,
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