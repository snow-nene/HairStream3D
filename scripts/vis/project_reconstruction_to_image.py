#!/usr/bin/env python3
"""将世界坐标中的重建网格直接投影到原始二维图片。"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon_3d.recon3D import load_calib
from lib.multiview_fusion import apply_image_similarity_to_calib


def project_visible_vertices(mesh, calib, height, width, depth_tolerance=0.003):
    """投影完整网格顶点，并用正交深度缓存保留最前表面。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    homogeneous = np.column_stack(
        [vertices, np.ones(len(vertices), dtype=np.float32)]
    )
    projected = homogeneous @ np.asarray(calib, dtype=np.float32).T
    ndc = projected[:, :2] / np.clip(projected[:, 3:4], 1e-8, None)
    px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
    py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    inside_ids = np.flatnonzero(inside)

    flat = py[inside_ids] * width + px[inside_ids]
    depth = projected[inside_ids, 2]
    zbuffer = np.full(height * width, -np.inf, dtype=np.float32)
    np.maximum.at(zbuffer, flat, depth)
    visible = depth >= zbuffer[flat] - float(depth_tolerance)
    visible_ids = inside_ids[visible]
    return px[visible_ids], py[visible_ids], projected[visible_ids, 2]


def render_projection(image, px, py, depth):
    """生成稠密深度着色投影、轮廓以及原图叠加。"""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    depth_image = np.full((height, width), np.nan, dtype=np.float32)
    mask[py, px] = 255
    depth_image[py, px] = depth

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    valid_depth = np.isfinite(depth_image)
    if valid_depth.any():
        lo, hi = np.percentile(depth_image[valid_depth], [2, 98])
        normalized = np.zeros_like(depth_image, dtype=np.uint8)
        normalized[valid_depth] = np.clip(
            (depth_image[valid_depth] - lo) / max(hi - lo, 1e-8) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        normalized = cv2.dilate(normalized, kernel, iterations=1)
    else:
        normalized = np.zeros((height, width), dtype=np.uint8)

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[mask == 0] = 0
    overlay = image.copy()
    mixed = cv2.addWeighted(image, 0.35, colored, 0.65, 0)
    overlay[mask > 0] = mixed[mask > 0]

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outline = image.copy()
    cv2.drawContours(outline, contours, -1, (0, 255, 255), 2)
    return colored, overlay, outline, mask


def main():
    parser = argparse.ArgumentParser(
        description="将重建三维网格无过滤地投影到原始 front 图片"
    )
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--calib", default=None)
    parser.add_argument(
        "--image_similarity",
        nargs=4,
        type=float,
        metavar=("ANGLE", "SCALE", "TX", "TY"),
        default=None,
        help="投影后的二维相似变换：角度、缩放、x/y 像素平移",
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    data_dir = ROOT / "results" / "multiview_data" / args.img_id
    mesh_path = Path(args.mesh) if args.mesh else (
        data_dir / "pixal3d" / "hair_mesh_aligned_best.obj"
    )
    calib_path = Path(args.calib) if args.calib else (
        data_dir / "maps" / "param" / "front.npy"
    )
    if args.image:
        image_path = Path(args.image)
    else:
        raw_path_file = data_dir / "raw_img_path.txt"
        image_path = (
            Path(raw_path_file.read_text(encoding="utf-8").strip())
            if raw_path_file.exists()
            else data_dir / "raw_img.png"
        )
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_dir / "pde_smoothness_experiments" / "reconstruction_projection"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取原图: {image_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"无效重建网格: {mesh_path}")

    # front.npy 的优化坐标系固定以 1024 为归一化基准；投影后的 NDC
    # 可以映射到任意实际图片尺寸。
    calib = load_calib(str(calib_path), loadSize=1024).numpy()
    if args.image_similarity is not None:
        calib = apply_image_similarity_to_calib(
            calib,
            *args.image_similarity,
            image_size=(image.shape[1], image.shape[0]),
        )
    px, py, depth = project_visible_vertices(
        mesh, calib, image.shape[0], image.shape[1]
    )
    colored, overlay, outline, mask = render_projection(image, px, py, depth)
    comparison = np.concatenate([image, colored, overlay, outline], axis=1)

    cv2.imwrite(str(out_dir / "mesh_depth_projection.png"), colored)
    cv2.imwrite(str(out_dir / "mesh_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / "mesh_outline.png"), outline)
    cv2.imwrite(str(out_dir / "mesh_projection_mask.png"), mask)
    cv2.imwrite(str(out_dir / "comparison.png"), comparison)
    print(f"mesh={mesh_path}")
    print(f"image={image_path}")
    print(f"visible_vertices={len(px)}, projected_pixels={(mask > 0).sum()}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
