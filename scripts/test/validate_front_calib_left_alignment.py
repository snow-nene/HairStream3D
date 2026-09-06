"""验证 Blender 真值相机矩阵能否把原始 GLB 精确投影到 left.png。

该测试读取渲染时保存的 ``camera_params/left.npz``，把原始 Pixel3D GLB
转换到 Blender 坐标系后投影，并用 ``left.png`` 的 alpha 通道作为真值轮廓。结果统一写入
``tests/outputs/calib_alignment/<image_id>/``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon_3d.extract_hair_mesh import load_glb_geometry


DEFAULT_IMAGE_ID = "0a1ba3dbefc8934ab60577c5c91f66a0"


def load_blender_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载 GLB，并从 glTF 的 Y-up 坐标转换到 Blender 的 Z-up 坐标。"""
    mesh = load_glb_geometry(path)
    gltf_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    blender_vertices = np.column_stack(
        [gltf_vertices[:, 0], -gltf_vertices[:, 2], gltf_vertices[:, 1]]
    )
    return blender_vertices, np.asarray(mesh.faces)


def project_with_blender_camera(
    vertices: np.ndarray,
    world_to_clip: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """按 Blender/OpenGL 约定投影，并将左下原点的 NDC 转为左上图像原点。"""
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    clip = homogeneous @ world_to_clip.T
    ndc = clip[:, :2] / np.clip(clip[:, 3:4], 1e-12, None)
    pixels = np.empty_like(ndc)
    pixels[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
    pixels[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * (height - 1)
    return pixels


def rasterize_mesh_mask(pixels: np.ndarray, faces: np.ndarray, width: int, height: int) -> np.ndarray:
    """将投影三角形光栅化为二值轮廓；轮廓验证不需要深度缓冲。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    triangles = np.rint(pixels[faces]).astype(np.int32)
    finite = np.isfinite(pixels[faces]).all(axis=(1, 2))
    for triangle in triangles[finite]:
        cv2.fillConvexPoly(mask, triangle, 255, lineType=cv2.LINE_8)
    return mask


def compute_alignment_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """计算轮廓 IoU、Dice 以及双向边界距离（像素）。"""
    pred = predicted > 0
    truth = target > 0
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    denom = int(pred.sum() + truth.sum())

    pred_edge = cv2.Canny(predicted, 100, 200) > 0
    truth_edge = cv2.Canny(target, 100, 200) > 0
    dist_to_truth = cv2.distanceTransform((~truth_edge).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate([dist_to_truth[pred_edge], dist_to_pred[truth_edge]])
    if distances.size == 0:
        distances = np.array([np.inf], dtype=np.float32)

    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / denom if denom else 1.0,
        "boundary_mean_px": float(np.mean(distances)),
        "boundary_p95_px": float(np.percentile(distances, 95)),
        "predicted_area_px": int(pred.sum()),
        "target_area_px": int(truth.sum()),
    }


def make_overlay(image_bgra: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """生成绿=重合、红=只预测、蓝=只真值的诊断叠加图。"""
    canvas = image_bgra[:, :, :3].copy()
    pred = predicted > 0
    truth = target > 0
    colors = np.zeros_like(canvas)
    colors[np.logical_and(pred, truth)] = (0, 220, 0)
    colors[np.logical_and(pred, ~truth)] = (0, 0, 255)
    colors[np.logical_and(~pred, truth)] = (255, 0, 0)
    changed = np.logical_or(pred, truth)
    canvas[changed] = cv2.addWeighted(canvas, 0.35, colors, 0.65, 0)[changed]
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--view", default="left")
    parser.add_argument("--camera-path", type=Path, default=None)
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--min-iou", type=float, default=0.95)
    parser.add_argument("--max-boundary-p95", type=float, default=2.0)
    args = parser.parse_args()

    data_dir = args.data_dir or ROOT / "results" / "multiview_data" / args.image_id
    output_dir = args.output_dir or ROOT / "tests" / "outputs" / "calib_alignment" / args.image_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = data_dir / "blender_renders" / f"{args.view}.png"
    camera_path = args.camera_path or data_dir / "blender_renders" / "camera_params" / f"{args.view}.npz"
    mesh_path = args.mesh_path or data_dir / "pixal3d" / f"{args.image_id}.glb"
    for path in (image_path, camera_path, mesh_path):
        if not path.exists():
            raise FileNotFoundError(path)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"渲染图必须是带 alpha 的 RGBA 图像: {image_path}")
    height, width = image.shape[:2]
    target_mask = np.where(image[:, :, 3] > 0, 255, 0).astype(np.uint8)

    camera = np.load(camera_path)
    world_to_clip = np.asarray(camera["world_to_clip"], dtype=np.float64)
    vertices, faces = load_blender_mesh(mesh_path)
    pixels = project_with_blender_camera(vertices, world_to_clip, width, height)
    mask = rasterize_mesh_mask(pixels, faces, width, height)
    metrics = compute_alignment_metrics(mask, target_mask)
    passed = metrics["iou"] >= args.min_iou and metrics["boundary_p95_px"] <= args.max_boundary_p95
    cv2.imwrite(str(output_dir / f"{args.view}_camera_mask.png"), mask)
    cv2.imwrite(str(output_dir / f"{args.view}_camera_overlay.png"), make_overlay(image, mask, target_mask))
    np.save(output_dir / f"{args.view}_world_to_clip.npy", world_to_clip)
    cv2.imwrite(str(output_dir / "target_alpha.png"), target_mask)

    report = {
        "image_id": args.image_id,
        "definition": "perfect := IoU >= min_iou 且双向边界距离 P95 <= max_boundary_p95",
        "thresholds": {"min_iou": args.min_iou, "max_boundary_p95_px": args.max_boundary_p95},
        "view": args.view,
        "passed": passed,
        "metrics": metrics,
        "inputs": {"image": str(image_path), "camera": str(camera_path), "raw_glb": str(mesh_path)},
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"诊断结果: {output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
