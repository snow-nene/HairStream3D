#!/usr/bin/env python3
"""
Step 1: 批量生成头发 mask

对输入目录下所有图片递归执行：
  1. pad+resize 到 512×512
  2. 生成 hair mask（seg）与黑底头发抠图（hair）

后端（--seg_backend）:
  sam3（默认）: 文本提示 SAM3，通过 subprocess 调用 ext/sam3 独立 pixi 环境
                中的 scripts/infer_2d/sam3_seg_worker.py
  sam        : 旧版 SAM ViT-H 点提示（fallback）

输出到 <output_dir>/img、<output_dir>/seg、<output_dir>/hair

Usage:
    PYTHONPATH=. pixi run python scripts/train/batch_generate_masks.py \
        --input_dir datasets/0002.mqset \
        --output_dir datasets/pretrain
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from segment_anything import SamPredictor, sam_model_registry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAM3_WORKER = os.path.join(ROOT, "scripts", "infer_2d", "sam3_seg_worker.py")


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


def run_sam3_worker(manifest_path, args):
    sam3_root = os.path.abspath(args.sam3_root)
    sam3_python = os.path.join(sam3_root, ".pixi", "envs", "default", "bin", "python")
    checkpoint = os.path.abspath(args.checkpoint_sam3)
    if not os.path.isfile(sam3_python):
        raise RuntimeError(
            f"找不到 SAM3 独立环境解释器: {sam3_python}\n"
            f"请先在 {sam3_root} 执行 `pixi install`，或改用 --seg_backend sam")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            f"找不到 SAM3 checkpoint: {checkpoint}\n"
            f"请用 --checkpoint_sam3 指定，或改用 --seg_backend sam")
    cmd = [
        sam3_python, SAM3_WORKER,
        "--manifest", manifest_path,
        "--checkpoint", checkpoint,
        "--hair_prompts", *[str(p) for p in args.sam3_hair_prompts],
        "--body_prompt", "",   # 训练数据不需要 body mask
        "--person_selection", "largest",   # 训练图可能含多人，只保留主角人物
        "--conf_threshold", str(args.sam3_conf_threshold),
        "--mask_channels", "1",
        "--device", args.device,
    ]
    print("running SAM3 worker:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Batch hair mask generation')
    parser.add_argument('--input_dir', default='datasets/0002.mqset')
    parser.add_argument('--output_dir', default='datasets/pretrain')
    parser.add_argument('--seg_backend', choices=['sam3', 'sam'], default='sam3',
                        help='sam3=文本提示 SAM3(默认), sam=旧版点提示 SAM ViT-H')
    parser.add_argument('--sam3_root', default=os.path.join(ROOT, 'ext', 'sam3'))
    parser.add_argument('--checkpoint_sam3',
                        default=os.path.join(ROOT, 'ext', 'sam3', 'checkpoints',
                                             'facebook', 'sam3.1', 'sam3.1_multiplex.pt'))
    parser.add_argument('--sam3_conf_threshold', type=float, default=0.3)
    parser.add_argument('--sam3_hair_prompts', nargs='+',
                        default=["hair", "ponytail hair and twin tails",
                                 "bangs and hair on the front"])
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

    # ── 收集图片并确定待处理列表 ──
    image_paths = collect_images(args.input_dir)
    if args.max_samples > 0:
        image_paths = image_paths[:args.max_samples]
    print(f"Found {len(image_paths)} images to process")

    todo = []   # (img_path, out_img, out_seg, out_hair)
    skipped = 0
    for img_path in image_paths:
        h = file_hash(img_path)
        out_img = os.path.join(args.output_dir, 'img', f'{h}.png')
        out_seg = os.path.join(args.output_dir, 'seg', f'{h}.png')
        out_hair = os.path.join(args.output_dir, 'hair', f'{h}.png')
        if os.path.exists(out_img) and os.path.exists(out_seg) and os.path.exists(out_hair):
            skipped += 1
            continue
        todo.append((img_path, out_img, out_seg, out_hair))

    # ── 统一 pad+resize 落盘 ──
    for img_path, out_img, out_seg, out_hair in tqdm(todo, desc="pad+resize"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        cv2.imwrite(out_img, pad_and_resize(image, width=args.width))

    if args.seg_backend == 'sam3':
        # ── SAM3: 一次性 worker 跑完所有 seg，再补写 hair 抠图 ──
        manifest = os.path.join(args.output_dir, '_sam3_manifest.json')
        items = [{"input": out_img, "seg_out": out_seg, "body_out": None}
                 for _, out_img, out_seg, _ in todo]
        with open(manifest, 'w') as f:
            json.dump({"items": items}, f, indent=2)
        try:
            run_sam3_worker(manifest, args)
        finally:
            if os.path.exists(manifest):
                os.remove(manifest)
        for _, out_img, out_seg, out_hair in tqdm(todo, desc="hair cutout"):
            mask = cv2.imread(out_seg, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(out_img)
            if mask is None or rgb is None:
                continue
            mask_f = (mask > 127).astype(np.float32)[:, :, None]
            cv2.imwrite(out_hair, (rgb.astype(np.float32) * mask_f).astype(np.uint8))
    else:
        # ── 旧版 SAM ViT-H 点提示 ──
        print(f"Loading SAM ({args.model_type_sam}) from {args.checkpoint_sam} ...")
        sam = sam_model_registry[args.model_type_sam](checkpoint=args.checkpoint_sam)
        sam.to(device=args.device)
        predictor = SamPredictor(sam)

        for img_path, out_img, out_seg, out_hair in tqdm(todo, desc="SAM masks"):
            resized = cv2.imread(out_img)
            if resized is None:
                continue
            predictor.set_image(resized)

            body_masks, _, _ = predictor.predict(
                point_coords=np.array([[255, 255]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

            body_mask_2d = body_masks[:, :, 255] if body_masks.shape[0] == 3 else body_masks[0]
            hair_pt_x = np.where(np.sum(body_mask_2d, axis=0) > 0)[0]
            if len(hair_pt_x) == 0:
                hair_pt = np.array([[255, 260]])
            else:
                hair_pt = np.array([[255, hair_pt_x[0] + 5]])

            pts = np.concatenate([hair_pt, np.array([[255, 255]])], axis=0)
            hair_masks, _, _ = predictor.predict(
                point_coords=pts,
                point_labels=np.array([1, 0]),
                multimask_output=False,
            )

            if hair_masks.shape[0] == 3 and hair_masks.ndim == 3:
                mask_2d = hair_masks.transpose(1, 2, 0)
                mask_bin = ((mask_2d[:, :, 0] + mask_2d[:, :, 1] + mask_2d[:, :, 2]) > 0)
            elif hair_masks.ndim == 3 and hair_masks.shape[0] == 1:
                mask_bin = (hair_masks[0] > 0)
            else:
                mask_bin = (hair_masks > 0)
            mask_u8 = (mask_bin.astype(np.uint8) * 255)
            cv2.imwrite(out_seg, mask_u8)

            mask_f = (mask_u8 / 255.0)[:, :, None]
            cv2.imwrite(out_hair, (resized.astype(np.float32) * mask_f).astype(np.uint8))

    print(f"\nDone: {len(todo)} processed, {skipped} skipped (already exist)")


if __name__ == '__main__':
    main()
