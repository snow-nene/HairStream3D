#!/usr/bin/env python
"""Visualize directions encoded in existing multi-view strand maps.

Outputs a quiver plot and an undirected streamline overlay for each requested
view without running the img2strand model.
"""

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_VIEWS = ("front", "left", "right", "back")


def decode_strand_map(strand_map):
    """Decode HairStep RGB encoding into image-space directions."""
    strand_map = np.asarray(strand_map, dtype=np.float32)
    if strand_map.ndim != 3 or strand_map.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 strand map, got {strand_map.shape}")
    if strand_map.max() > 1.0:
        strand_map = strand_map / 255.0

    mask = strand_map[:, :, 0] > 0.1
    dx = 1.0 - 2.0 * strand_map[:, :, 2]
    dy = 2.0 * strand_map[:, :, 1] - 1.0
    magnitude = np.sqrt(dx * dx + dy * dy)
    valid = mask & (magnitude > 1e-4)
    dx = np.where(valid, dx / np.maximum(magnitude, 1e-8), 0.0)
    dy = np.where(valid, dy / np.maximum(magnitude, 1e-8), 0.0)
    return dx, dy, valid


def load_background(data_dir, view, shape):
    """Load the image corresponding to a strand-map view when available."""
    candidates = [data_dir / "flux_redrawn" / f"{view}.png"]
    if view == "front":
        raw_path_file = data_dir / "raw_img_path.txt"
        if raw_path_file.exists():
            raw_path = Path(raw_path_file.read_text(encoding="utf-8").strip())
            if raw_path:
                candidates.append(raw_path)
    candidates.append(data_dir / "blender_renders" / f"{view}.png")

    for path in candidates:
        if not path.exists():
            continue
        image = imageio.imread(path)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        image = image[:, :, :3]
        height, width = shape
        if image.shape[:2] != shape:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8)
    return np.full((*shape, 3), 245, dtype=np.uint8)


def save_quiver(background, dx, dy, mask, output_path, step):
    """Save sparse direction arrows over the source image."""
    height, width = mask.shape
    ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
    sampled = mask[ys, xs]
    canvas = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    arrow_length = max(step * 0.42, 4.0)
    thickness = max(1, int(round(step / 12.0)))
    for x, y, vx, vy in zip(
        xs[sampled],
        ys[sampled],
        dx[ys, xs][sampled],
        dy[ys, xs][sampled],
    ):
        half = 0.5 * arrow_length
        start = (int(round(x - half * vx)), int(round(y - half * vy)))
        end = (int(round(x + half * vx)), int(round(y + half * vy)))
        cv2.arrowedLine(
            canvas,
            start,
            end,
            (35, 35, 245),
            thickness=thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.28,
        )
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"无法保存箭头图: {output_path}")


def bilinear_direction(dx, dy, mask, x, y):
    """Sample and normalize the line field at a floating-point pixel."""
    height, width = mask.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
        return None
    if not (mask[y0, x0] and mask[y0, x1] and mask[y1, x0] and mask[y1, x1]):
        return None

    fx, fy = x - x0, y - y0
    weights = np.array([
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ])
    vx = np.dot(weights, [dx[y0, x0], dx[y0, x1], dx[y1, x0], dx[y1, x1]])
    vy = np.dot(weights, [dy[y0, x0], dy[y0, x1], dy[y1, x0], dy[y1, x1]])
    norm = np.hypot(vx, vy)
    if norm < 1e-4:
        return None
    return vx / norm, vy / norm


def trace_half(dx, dy, mask, seed, sign, step_size, max_steps):
    """Trace one half of an undirected streamline with sign continuity."""
    x, y = seed
    initial = bilinear_direction(dx, dy, mask, x, y)
    if initial is None:
        return []
    vx, vy = initial[0] * sign, initial[1] * sign
    points = []
    for _ in range(max_steps):
        midpoint = (x + 0.5 * step_size * vx, y + 0.5 * step_size * vy)
        sampled = bilinear_direction(dx, dy, mask, *midpoint)
        if sampled is None:
            break
        next_vx, next_vy = sampled
        if next_vx * vx + next_vy * vy < 0.0:
            next_vx, next_vy = -next_vx, -next_vy
        x += step_size * next_vx
        y += step_size * next_vy
        if bilinear_direction(dx, dy, mask, x, y) is None:
            break
        points.append((x, y))
        vx, vy = next_vx, next_vy
    return points


def save_streamlines(
    background,
    dx,
    dy,
    mask,
    output_path,
    seed_step,
    trace_step,
    max_length,
    line_width,
):
    """Save bidirectional streamlines over the source image."""
    height, width = mask.shape
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell_size = max(seed_step // 2, 1)
    covered = np.zeros(
        ((height + cell_size - 1) // cell_size,
         (width + cell_size - 1) // cell_size),
        dtype=bool,
    )
    max_steps = max(1, int(max_length / trace_step))

    for y in range(seed_step // 2, height, seed_step):
        for x in range(seed_step // 2, width, seed_step):
            if not mask[y, x] or covered[y // cell_size, x // cell_size]:
                continue
            forward = trace_half(dx, dy, mask, (x, y), 1.0, trace_step, max_steps)
            backward = trace_half(dx, dy, mask, (x, y), -1.0, trace_step, max_steps)
            points = list(reversed(backward)) + [(float(x), float(y))] + forward
            if len(points) < 4:
                continue
            for px, py in points:
                row, column = int(py // cell_size), int(px // cell_size)
                if 0 <= row < covered.shape[0] and 0 <= column < covered.shape[1]:
                    covered[row, column] = True
            draw.line(points, fill=(235, 35, 35, 190), width=line_width)

    base = Image.fromarray(background).convert("RGBA")
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化已有多视图 strand_map 的二维方向场"
    )
    parser.add_argument("--img_id", required=True, help="multiview_data 的 Image ID")
    parser.add_argument(
        "--views", nargs="+", default=["front"], help="需要可视化的任意视角集合"
    )
    parser.add_argument(
        "--all-available", action="store_true", help="自动处理 front/left/right/back 中存在的视角"
    )
    parser.add_argument("--quiver-step", type=int, default=20, help="箭头采样间隔")
    parser.add_argument("--seed-step", type=int, default=24, help="流线种子间隔")
    parser.add_argument("--trace-step", type=float, default=2.0, help="流线积分步长")
    parser.add_argument("--max-length", type=float, default=140.0, help="单侧最大流线长度")
    parser.add_argument("--line-width", type=int, default=2, help="流线宽度")
    parser.add_argument(
        "--render-scale",
        type=int,
        default=4,
        help="相对 strand map 的输出倍率（默认 4，即 512→2048）",
    )
    parser.add_argument("--out_dir", default=None, help="自定义输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = ROOT / "results" / "multiview_data" / args.img_id
    strand_dir = data_dir / "maps" / "strand_map"
    output_dir = (
        Path(args.out_dir)
        if args.out_dir
        else data_dir / "visualizations" / "strand_direction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = DEFAULT_VIEWS if args.all_available else tuple(args.views)
    processed = 0
    for view in requested:
        strand_path = strand_dir / f"{view}.png"
        if not strand_path.exists():
            print(f"[跳过] 缺少 {view} strand map: {strand_path}")
            continue
        strand_map = imageio.imread(strand_path)
        dx, dy, mask = decode_strand_map(strand_map)
        if args.render_scale < 1:
            raise ValueError("--render-scale 必须大于等于 1")
        height, width = mask.shape
        render_shape = (
            height * args.render_scale,
            width * args.render_scale,
        )
        dx = cv2.resize(
            dx, render_shape[::-1], interpolation=cv2.INTER_LINEAR
        )
        dy = cv2.resize(
            dy, render_shape[::-1], interpolation=cv2.INTER_LINEAR
        )
        magnitude = np.sqrt(dx * dx + dy * dy)
        dx /= np.maximum(magnitude, 1e-8)
        dy /= np.maximum(magnitude, 1e-8)
        mask = cv2.resize(
            mask.astype(np.uint8),
            render_shape[::-1],
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        dx[~mask] = 0.0
        dy[~mask] = 0.0
        background = load_background(data_dir, view, render_shape)
        quiver_path = output_dir / f"{view}_quiver.png"
        streamline_path = output_dir / f"{view}_streamlines.png"
        save_quiver(
            background,
            dx,
            dy,
            mask,
            quiver_path,
            args.quiver_step * args.render_scale,
        )
        save_streamlines(
            background,
            dx,
            dy,
            mask,
            streamline_path,
            args.seed_step * args.render_scale,
            args.trace_step * args.render_scale,
            args.max_length * args.render_scale,
            args.line_width * args.render_scale,
        )
        processed += 1
        print(
            f"[{view}] size={render_shape[1]}x{render_shape[0]}, "
            f"valid={int(mask.sum())} → "
            f"{quiver_path.name}, {streamline_path.name}"
        )

    if not processed:
        raise RuntimeError(f"没有找到可处理的 strand map: {strand_dir}")
    print(f"完成：{processed} 个视角 → {output_dir}")


if __name__ == "__main__":
    main()
