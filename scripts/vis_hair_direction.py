#!/usr/bin/env python
"""使用 HairStep_ours 的 img2strand 模型对 test_hairs 做推理，
输出方向场 RGB 图 + 流线叠加可视化。

参考 ReChannel/scripts/vis_hair_direction.py 的可视化方式。

Usage:
  python scripts/vis_hair_direction.py
  python scripts/vis_hair_direction.py --backbone hrnet --checkpoint checkpoints/pretrain/hrnet_w32/img2strand_hrnet_w32_epoch_10.pth
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.model.img2hairstep.model_factory import create_img2strand_model
from lib.options import BaseOptions


# ─── 路径 ───────────────────────────────────────────────────────────
TEST_HAIRS = REPO / "test_hairs"
RAW_DIR = TEST_HAIRS / "raw_pictures"
SEG_DIR = TEST_HAIRS / "seg"
DEFAULT_CKPT = REPO / "checkpoints" / "img2hairstep" / "img2strand.pth"
OUT_DIR = REPO / "test_hairs_out"

# ─── 参数默认值 ────────────────────────────────────────────────────
RESOLUTION = 512
SCALE = 4
OVERLAY_STEP = 32
OVERLAY_MAX_LEN = 120
OVERLAY_W = 2
TRACE_STEP = 3


# ─── 方向场 → RGB ─────────────────────────────────────────────────
def _direction_to_rgb(dx: np.ndarray, dy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """R=mask, G=(dy+1)/2, B=(-dx+1)/2"""
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)
    h, w = dx.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m = mask.clip(0, 1)
    rgb[:, :, 0] = (m * 255).astype(np.uint8)
    rgb[:, :, 1] = ((dy + 1.0) / 2.0 * 255 * m).astype(np.uint8)
    rgb[:, :, 2] = ((-dx + 1.0) / 2.0 * 255 * m).astype(np.uint8)
    return rgb


# ─── 流线叠加 ─────────────────────────────────────────────────────
def _streamline_overlay(
    rgb: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    mask: np.ndarray, seed_step: int = 32,
    trace_step: float = 3.0, max_steps: int = 80,
    color: tuple = (220, 40, 40, 150), width: int = 2,
) -> np.ndarray:
    H, W = mask.shape

    def _sample(x: float, y: float):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
            return 0.0, 0.0, False
        if not (mask[y0, x0] and mask[y0, x1] and mask[y1, x0] and mask[y1, x1]):
            return 0.0, 0.0, False
        fx, fy = x - x0, y - y0
        vx = ((1 - fx) * dx[y0, x0] + fx * dx[y0, x1]) * (1 - fy) \
           + ((1 - fx) * dx[y1, x0] + fx * dx[y1, x1]) * fy
        vy = ((1 - fx) * dy[y0, x0] + fx * dy[y0, x1]) * (1 - fy) \
           + ((1 - fx) * dy[y1, x0] + fx * dy[y1, x1]) * fy
        m2 = vx * vx + vy * vy
        if m2 < 1e-4:
            return 0.0, 0.0, False
        inv = 1.0 / np.sqrt(m2)
        return vx * inv, vy * inv, True

    def _trace(sx: float, sy: float, sign: float):
        pts = []
        cx, cy = sx, sy
        vx, vy, ok = _sample(cx, cy)
        if not ok:
            return pts
        vx, vy = vx * sign, vy * sign
        for _ in range(max_steps):
            half = trace_step * 0.5
            mx, my = cx + vx * half, cy + vy * half
            nvx, nvy, ok = _sample(mx, my)
            if not ok:
                break
            if nvx * vx + nvy * vy < 0:
                nvx, nvy = -nvx, -nvy
            cx += nvx * trace_step
            cy += nvy * trace_step
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                break
            xi, yi = int(cx), int(cy)
            if 0 <= xi < W and 0 <= yi < H and not mask[yi, xi]:
                break
            pts.append((cx, cy))
            vx, vy = nvx, nvy
        return pts

    seeds_y, seeds_x = np.where(mask[seed_step // 2::seed_step, seed_step // 2::seed_step])
    seeds_y = seeds_y * seed_step + seed_step // 2
    seeds_x = seeds_x * seed_step + seed_step // 2

    cell_size = seed_step // 2
    grid_h = (H + cell_size - 1) // cell_size
    grid_w = (W + cell_size - 1) // cell_size
    covered = np.zeros((grid_h, grid_w), dtype=bool)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for sy, sx in zip(seeds_y, seeds_x):
        gi, gj = int(sy // cell_size), int(sx // cell_size)
        if covered[gi, gj]:
            continue
        fwd = _trace(float(sx), float(sy), +1)
        bwd = _trace(float(sx), float(sy), -1)
        if len(fwd) + len(bwd) < 3:
            continue
        all_pts = list(reversed(bwd)) + [(float(sx), float(sy))] + fwd
        for px, py in all_pts:
            ci, cj = int(py // cell_size), int(px // cell_size)
            if 0 <= ci < grid_h and 0 <= cj < grid_w:
                covered[ci, cj] = True
        draw.line(all_pts, fill=color, width=width)

    rgb_rgba = Image.fromarray(rgb).convert("RGBA")
    result = Image.alpha_composite(rgb_rgba, overlay)
    return np.array(result.convert("RGB"))


def _upscale(arr: np.ndarray, size: int, method=Image.LANCZOS) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size), method))


def _pad_and_resize(img: np.ndarray, width: int = 512) -> np.ndarray:
    """正方形 padding + resize"""
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)), constant_values=0) if img.ndim == 3 else np.pad(img, ((pad, pad), (0, 0)), constant_values=0)
    else:
        pad = (w - h) // 2
        padded = np.pad(img, ((pad, pad), (0, 0), (0, 0)), constant_values=0) if img.ndim == 3 else np.pad(img, ((0, 0), (pad, pad)), constant_values=0)
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(padded, (width, width), interpolation=interp)
    return resized


def _find_image_pairs(img_dir, seg_dir):
    """发现 (raw_image, seg_mask) 对"""
    pairs = []
    for img_path in sorted(Path(img_dir).iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        stem = img_path.stem
        seg = Path(seg_dir)
        mask_path = None
        for cand in [
            seg / f"{stem}_mask.png",
            seg / f"{stem}_mask.jpg",
            seg / f"{stem}.png",
            seg / f"{stem}.jpg",
        ]:
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            print(f"  [SKIP] No mask found for {img_path.name}")
            continue
        pairs.append((img_path, mask_path))
    return pairs


def main():
    import argparse

    # 先 parse 我们自己的参数（不传给 BaseOptions）
    ap = argparse.ArgumentParser(description="HairStep img2strand 推理 + 可视化")
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                    help=f"模型 checkpoint")
    ap.add_argument("--backbone", type=str, default="unet",
                    choices=["unet", "hrnet"])
    ap.add_argument("--hrnet-variant", type=str, default="hrnet_w32",
                    choices=["hrnet_w18", "hrnet_w32", "hrnet_w48"])
    ap.add_argument("--input-dir", type=str, default=str(RAW_DIR.parent),
                    help=f"输入目录（含 raw_pictures/ 和 seg/）")
    ap.add_argument("--output-dir", type=str, default=str(OUT_DIR),
                    help="输出目录")
    ap.add_argument("--step", type=int, default=OVERLAY_STEP)
    ap.add_argument("--max-len", type=int, default=OVERLAY_MAX_LEN)
    ap.add_argument("--line-width", type=int, default=OVERLAY_W)
    ap.add_argument("--scale", type=int, default=SCALE)
    ap.add_argument("--alpha", type=int, default=150)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-panel", action="store_true")
    args, remaining = ap.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = Path(args.input_dir) / "raw_pictures"
    if not img_dir.exists():
        img_dir = Path(args.input_dir) / "img"
    seg_dir = Path(args.input_dir) / "seg"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_pairs(img_dir, seg_dir)
    if not pairs:
        print(f"ERROR: 没有在 {img_dir} 和 {seg_dir} 里找到图片/mask 对")
        sys.exit(1)
    if not pairs:
        print("ERROR: 没有在 test_hairs/ 里找到图片/mask 对")
        sys.exit(1)
    print(f"找到 {len(pairs)} 张图片")

    # 构建 opt（让它只看 remaining，避免参数冲突）
    import copy
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]] + remaining  # 只保留剩余参数给 BaseOptions
    opt = BaseOptions().parse()
    sys.argv = old_argv  # 恢复

    opt.img2strand_backbone = args.backbone
    if args.backbone == "hrnet":
        opt.hrnet_variant = args.hrnet_variant

    print(f"加载 checkpoint: {args.ckpt}")
    model = create_img2strand_model(opt).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rows = []
    t0 = time.time()

    for idx, (img_path, mask_path) in enumerate(pairs):
        name = img_path.stem
        print(f"[{idx+1}/{len(pairs)}] {img_path.name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] 无法读取 {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            print(f"  [SKIP] 无法读取 mask {mask_path}")
            continue

        # pad + resize 到 512
        img_512 = _pad_and_resize(img_rgb, RESOLUTION)
        mask_512 = _pad_and_resize(mask_raw, RESOLUTION)
        mask_512 = (mask_512 > 127).astype(np.float32)

        # 推理
        img_tensor = torch.from_numpy(img_512.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            # UNet 返回单 tensor，HRNet 返回 (pred, aux_preds)
            pred = out[0] if isinstance(out, tuple) else out  # (1, 2, 512, 512)

        # 模型输出 2ch [0,1]: ch0=G=(dy+1)/2, ch1=B=(-dx+1)/2
        dy_raw = pred[0, 0].cpu().float().numpy()
        dx_raw = pred[0, 1].cpu().float().numpy()

        # 解码到 [-1, 1]
        dy = dy_raw * 2.0 - 1.0          # G: dy = val*2 - 1
        dx = 1.0 - dx_raw * 2.0          # B: dx = 1 - val*2
        dx = dx.clip(-1, 1)
        dy = dy.clip(-1, 1)

        # 输出 1: 方向场 RGB
        dir_rgb = _direction_to_rgb(dx, dy, mask_512)
        dir_path = out_dir / f"{name}_direction.png"
        Image.fromarray(dir_rgb).save(dir_path)
        print(f"  → {dir_path.name}")

        # 上采样
        R = args.scale * RESOLUTION
        img_hi = _upscale(img_512, R, Image.LANCZOS)
        pd_dx = _upscale(dx, R, Image.BILINEAR)
        pd_dy = _upscale(dy, R, Image.BILINEAR)
        pd_mask = _upscale((mask_512 * 255).astype(np.uint8), R, Image.NEAREST).astype(bool)

        # 输出 2: 流线叠加
        overlay = None
        if not args.no_overlay:
            max_steps = int(args.max_len / TRACE_STEP)
            overlay = _streamline_overlay(
                img_hi, pd_dx.clip(-1, 1), pd_dy.clip(-1, 1), pd_mask,
                seed_step=args.step,
                trace_step=TRACE_STEP,
                max_steps=max_steps,
                color=(220, 40, 40, args.alpha),
                width=args.line_width,
            )
            overlay_path = out_dir / f"{name}_overlay.png"
            Image.fromarray(overlay).save(overlay_path)
            print(f"  → {overlay_path.name}")

        # 面板行
        dir_rgb_hi = _upscale(dir_rgb, R, Image.NEAREST)
        if not args.no_panel:
            rows.append(np.concatenate(
                [img_hi, dir_rgb_hi,
                 overlay if not args.no_overlay else dir_rgb_hi],
                axis=1))

    # 拼接面板
    if rows and not args.no_panel:
        panel = np.concatenate(rows, axis=0)
        panel_path = out_dir / "panel.png"
        Image.fromarray(panel).save(panel_path)
        print(f"  → {panel_path.name}")

    elapsed = time.time() - t0
    print(f"\n完成 — {len(pairs)} 张图片, 耗时 {elapsed:.1f}s → {out_dir}/")


if __name__ == "__main__":
    main()
