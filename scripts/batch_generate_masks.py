#!/usr/bin/env python3
"""
Step 1: 批量生成头发 mask（SAM 抠图）

对 datasets/0002.mqset/ 下所有图片递归执行：
  1. pad+resize 到 512×512
  2. SAM 生成 hair mask
输出到 datasets/pretrain/img/ 和 datasets/pretrain/seg/

Usage:
    PYTHONPATH=. pixi run python scripts/batch_generate_masks.py \
        --input_dir datasets/0002.mqset \
        --output_dir datasets/pretrain
"""
import os
import sys
import hashlib
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from segment_anything import SamPredictor, sam_model_registry


# ── 复用 img2masks.py 的工具函数 ──
def pad_and_resize(img, width=512):
    img = np.array(img)
    if img.shape[0] > img.shape[1]:
        img_pad = np.zeros((img.shape[0], int((img.shape[0] - img.shape[1]) / 2), img.shape[2]),
                           dtype=img.dtype)
        padded_img = np.concatenate((img_pad, img, img_pad), axis=1)
    else:
        img_pad = np.zeros((int((img.shape[1] - img.shape[0]) / 2), img.shape[1], img.shape[2]),
                           dtype=img.dtype)
        padded_img = np.concatenate((img_pad, img, img_pad), axis=0)
    padded_img = resize(padded_img, (width, width), preserve_range=True).astype(np.uint8)
    return padded_img


def file_hash(path):
    """返回文件内容的 MD5 前 16 字符作为短名"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def collect_images(input_dir, exts=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    """递归收集所有图片文件"""
    files = []
    for root, _, fnames in os.walk(input_dir):
        for f in fnames:
            if f.lower().endswith(exts) and not f.startswith('.'):
                files.append(os.path.join(root, f))
    return files


def main():
    parser = argparse.ArgumentParser(description='Batch SAM mask generation')
    parser.add_argument('--input_dir', default='datasets/0002.mqset')
    parser.add_argument('--output_dir', default='datasets/pretrain')
    parser.add_argument('--checkpoint_sam', default='./checkpoints/SAM-models/sam_vit_h_4b8939.pth')
    parser.add_argument('--model_type_sam', default='vit_h')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit to N images (0 = all). Use for smoke test.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'seg'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'hair'), exist_ok=True)

    # ── 加载 SAM ──
    print(f"Loading SAM ({args.model_type_sam}) from {args.checkpoint_sam} ...")
    sam = sam_model_registry[args.model_type_sam](checkpoint=args.checkpoint_sam)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # ── 收集图片 ──
    image_paths = collect_images(args.input_dir)
    if args.max_samples > 0:
        image_paths = image_paths[:args.max_samples]
    print(f"Found {len(image_paths)} images to process")

    success, skipped = 0, 0
    for img_path in tqdm(image_paths, desc="SAM masks"):
        # 读取原图
        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        # pad + resize
        resized = pad_and_resize(image, width=args.width)

        # 生成唯一文件名
        h = file_hash(img_path)
        out_img_path = os.path.join(args.output_dir, 'img', f'{h}.png')
        out_seg_path = os.path.join(args.output_dir, 'seg', f'{h}.png')
        out_hair_path = os.path.join(args.output_dir, 'hair', f'{h}.png')

        # 如果三份都已存在则跳过
        if (os.path.exists(out_img_path) and
                os.path.exists(out_seg_path) and
                os.path.exists(out_hair_path)):
            success += 1
            continue

        # 保存 resized 全图
        cv2.imwrite(out_img_path, resized)

        # SAM 推理
        predictor.set_image(resized)

        # — 身体 mask（屏幕中心 255,255 点）—
        body_masks, _, _ = predictor.predict(
            point_coords=np.array([[255, 255]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        # — 头发 mask（身体 mask 上方边缘 + 5px）—
        body_mask_2d = body_masks[:, :, 255] if body_masks.shape[0] == 3 else body_masks[0]
        hair_pt_x = np.where(np.sum(body_mask_2d, axis=0) > 0)[0]
        if len(hair_pt_x) == 0:
            hair_pt = np.array([[255, 260]])
        else:
            hair_pt_x_val = hair_pt_x[0]
            hair_pt = np.array([[255, hair_pt_x_val + 5]])

        pts = np.concatenate([hair_pt, np.array([[255, 255]])], axis=0)
        hair_masks, _, _ = predictor.predict(
            point_coords=pts,
            point_labels=np.array([1, 0]),
            multimask_output=False,
        )

        # 解析为二值 mask (H, W) 0/255
        if hair_masks.shape[0] == 3 and hair_masks.ndim == 3:
            mask_2d = hair_masks.transpose(1, 2, 0)
            mask_bin_3c = ((mask_2d[:, :, 0] + mask_2d[:, :, 1] + mask_2d[:, :, 2]) > 0)
        elif hair_masks.ndim == 3 and hair_masks.shape[0] == 1:
            mask_bin_3c = (hair_masks[0] > 0)
        else:
            mask_bin_3c = (hair_masks > 0)

        mask_bin = (mask_bin_3c.astype(np.uint8) * 255)

        # ── 保存 mask ──
        cv2.imwrite(out_seg_path, mask_bin)

        # ── 保存抠出的头发（RGB × mask，黑色背景）──
        mask_float = (mask_bin / 255.0)[:, :, None]
        hair_only = (resized.astype(np.float32) * mask_float).astype(np.uint8)
        cv2.imwrite(out_hair_path, hair_only)

        success += 1

    print(f"\nDone: {success} succeeded, {skipped} skipped/failed")


if __name__ == '__main__':
    main()