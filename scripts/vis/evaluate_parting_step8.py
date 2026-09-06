#!/usr/bin/env python3
"""Step 8：在原图与多视图上评估发缝可见性、方向、轮廓和遮挡。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import read_ordered_strands
from scripts.recon_3d.couple_parting_outer_strands import resample_polyline
from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration
from scripts.recon_3d.solve_parting_bank_only_field import calibration_from_param


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
VIEWS = ("front", "left", "right", "back")
MODELS = ("v30", "v4_groom", "step7")
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


@dataclass(frozen=True)
class VisibilityConfig:
    """Step 8 软件光栅化与相对门禁参数。"""

    image_size: int = 512
    physical_sampling_m: float = 0.002
    head_visibility_tolerance_m: float = 0.002
    occupancy_dilation_px: int = 1
    direction_error_allowance_deg: float = 2.0
    silhouette_iou_allowance: float = 0.02
    behind_head_fraction_allowance: float = 0.005


def load_step8_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 7 已人工接受，并读取三组发丝和四视角输入。"""

    step7_report = json.loads(args.step7_report.read_text(encoding="utf-8"))
    if not step7_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 7 数值门禁未通过")
    if step7_report.get("human_review_status") != "accepted":
        raise RuntimeError("Step 7 尚未被人工标记为 accepted")
    models = {
        "v30": read_ordered_strands(args.v30_hair),
        "v4_groom": read_ordered_strands(args.v4_groom_hair),
        "step7": read_ordered_strands(args.step7_hair),
    }
    images = {}
    seg = {}
    strand_maps = {}
    for view in VIEWS:
        image_path = args.front_image if view == "front" else args.data_root / "flux_redrawn" / f"{view}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(args.data_root / "maps/seg" / f"{view}.png"), cv2.IMREAD_GRAYSCALE)
        strand_map = cv2.imread(str(args.data_root / "maps/strand_map" / f"{view}.png"), cv2.IMREAD_COLOR)
        if image is None or mask is None or strand_map is None:
            raise FileNotFoundError(f"缺少 {view} 的图像、seg 或 strand_map")
        images[view] = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        seg[view] = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
        strand_maps[view] = cv2.resize(strand_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    parting_mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if parting_mask is None:
        raise FileNotFoundError(f"缺少发缝 mask: {args.parting_mask}")
    parting_mask = cv2.resize(parting_mask, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    head = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not head.has_vertices() or not head.has_triangles():
        raise ValueError("头模为空")
    return {
        "step7_report": step7_report,
        "models": models,
        "images": images,
        "seg": seg,
        "strand_maps": strand_maps,
        "parting_mask": parting_mask,
        "head": head,
    }


def load_view_camera(args: argparse.Namespace, view: str) -> dict[str, np.ndarray]:
    """加载 HairStep world 到像素 NDC 的正交相机。"""

    if view == "front":
        calib = calibration_from_param(args.front_calib)
        depth_calib = calibration_from_param(args.front_depth_calib)
        calib[2] = depth_calib[2]
    else:
        calib, _ = load_blender_view_calibration(str(args.data_root), view)
        calib = np.asarray(calib, dtype=np.float64)
    linear = calib[:3, :3]
    toward_camera = np.linalg.solve(linear, np.array([0.0, 0.0, 1.0]))
    toward_camera /= max(np.linalg.norm(toward_camera), 1e-12)
    return {"calib": calib, "toward_camera": toward_camera}


def project_view(points: np.ndarray, calib: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """投影三维点，返回像素坐标和 NDC z。"""

    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ np.asarray(calib, dtype=np.float64).T
    ndc = clip[:, :3] / np.maximum(np.abs(clip[:, 3:4]), 1e-12) * np.sign(clip[:, 3:4])
    pixels = np.column_stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * (image_size - 1),
            (ndc[:, 1] + 1.0) * 0.5 * (image_size - 1),
        ]
    )
    return pixels, ndc[:, 2]


def build_head_depth_map(
    head: o3d.geometry.TriangleMesh,
    calib: np.ndarray,
    toward_camera: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用正交射线得到每个像素到头模首交点的距离。"""

    y, x = np.mgrid[0:image_size, 0:image_size]
    ndc_x = 2.0 * x.reshape(-1) / max(image_size - 1, 1) - 1.0
    ndc_y = 2.0 * y.reshape(-1) / max(image_size - 1, 1) - 1.0
    clip = np.column_stack([ndc_x, ndc_y, np.zeros_like(ndc_x), np.ones_like(ndc_x)])
    inverse = np.linalg.inv(np.asarray(calib, dtype=np.float64))
    base_h = clip @ inverse.T
    base = base_h[:, :3] / base_h[:, 3:4]
    extent = np.linalg.norm(np.asarray(head.get_axis_aligned_bounding_box().get_extent()))
    ray_length = max(float(extent) * 3.0, 1.0)
    origins = base + toward_camera[None, :] * ray_length
    direction = -toward_camera
    directions = np.broadcast_to(direction, origins.shape).copy()
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head))
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(image_size, image_size)
    return hit, origins.reshape(image_size, image_size, 3), direction


def collect_projected_segments(
    strands: list[np.ndarray],
    calib: np.ndarray,
    image_size: int,
    physical_sampling_m: float,
) -> dict[str, np.ndarray]:
    """先统一物理采样间距，再把可变长发束展开为投影线段。"""

    starts = []
    ends = []
    for strand in strands:
        try:
            clean, _ = resample_polyline(strand, physical_sampling_m)
        except ValueError:
            continue
        starts.append(clean[:-1])
        ends.append(clean[1:])
    start = np.concatenate(starts)
    end = np.concatenate(ends)
    midpoint = 0.5 * (start + end)
    start_px, _ = project_view(start, calib, image_size)
    end_px, _ = project_view(end, calib, image_size)
    midpoint_px, _ = project_view(midpoint, calib, image_size)
    return {"start": start, "end": end, "midpoint": midpoint, "start_px": start_px, "end_px": end_px, "midpoint_px": midpoint_px}


def rasterize_hair_view(
    segments: dict[str, np.ndarray],
    head_depth: np.ndarray,
    ray_origins: np.ndarray,
    ray_direction: np.ndarray,
    config: VisibilityConfig,
) -> dict[str, np.ndarray | float | int]:
    """以线段中点做头模遮挡测试，并生成可见发丝占据图。"""

    midpoint_px = np.asarray(segments["midpoint_px"])
    px = np.rint(midpoint_px[:, 0]).astype(np.int32)
    py = np.rint(midpoint_px[:, 1]).astype(np.int32)
    inside = (px >= 0) & (px < config.image_size) & (py >= 0) & (py < config.image_size)
    point_ids = np.flatnonzero(inside)
    hair_distance = np.zeros(len(px), dtype=np.float64)
    hair_distance[point_ids] = np.sum(
        (np.asarray(segments["midpoint"])[point_ids] - ray_origins[py[point_ids], px[point_ids]])
        * ray_direction[None, :],
        axis=1,
    )
    head_distance = np.full(len(px), np.inf, dtype=np.float64)
    head_distance[point_ids] = head_depth[py[point_ids], px[point_ids]]
    visible = inside & (hair_distance <= head_distance + config.head_visibility_tolerance_m)
    behind = inside & np.isfinite(head_distance) & ~visible
    occupancy = np.zeros((config.image_size, config.image_size), dtype=np.uint8)
    occupancy[py[visible], px[visible]] = 255
    kernel_size = 2 * int(config.occupancy_dilation_px) + 1
    occupancy = cv2.dilate(occupancy, np.ones((kernel_size, kernel_size), np.uint8))
    return {
        "occupancy": occupancy > 0,
        "visible": visible,
        "inside": inside,
        "behind": behind,
        "px": px,
        "py": py,
        "behind_fraction": float(np.sum(behind) / max(np.sum(inside), 1)),
    }


def decode_target_direction(strand_map_bgr: np.ndarray, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """解码 RGB strand_map 的无向二维方向。"""

    rgb = strand_map_bgr[py, px][:, ::-1].astype(np.float64) / 255.0
    direction = np.column_stack([1.0 - 2.0 * rgb[:, 2], 2.0 * rgb[:, 1] - 1.0])
    norm = np.linalg.norm(direction, axis=1)
    valid = (rgb[:, 0] > 0.1) & (norm > 1e-6)
    direction /= np.maximum(norm[:, None], 1e-12)
    return direction, valid


def evaluate_model_view(
    segments: dict[str, np.ndarray],
    raster: dict[str, np.ndarray | float | int],
    seg: np.ndarray,
    strand_map: np.ndarray,
    parting_mask: np.ndarray | None,
) -> dict[str, object]:
    """计算方向误差、轮廓 IoU、头模遮挡和正面发缝可见率。"""

    occupancy = np.asarray(raster["occupancy"], dtype=bool)
    intersection = int(np.sum(occupancy & seg))
    union = int(np.sum(occupancy | seg))
    visible = np.asarray(raster["visible"], dtype=bool)
    px = np.asarray(raster["px"])[visible]
    py = np.asarray(raster["py"])[visible]
    projected_direction = np.asarray(segments["end_px"])[visible] - np.asarray(segments["start_px"])[visible]
    projected_norm = np.linalg.norm(projected_direction, axis=1)
    projected_direction /= np.maximum(projected_norm[:, None], 1e-12)
    target, target_valid = decode_target_direction(strand_map, px, py)
    direction_valid = target_valid & (projected_norm > 0.25)
    cosine = np.abs(np.sum(projected_direction[direction_valid] * target[direction_valid], axis=1))
    error = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    metrics: dict[str, object] = {
        "visible_segment_count": int(np.sum(visible)),
        "behind_head_segment_count": int(np.sum(raster["behind"])),
        "behind_head_fraction": float(raster["behind_fraction"]),
        "silhouette_iou": float(intersection / max(union, 1)),
        "silhouette_false_positive_fraction": float(np.sum(occupancy & ~seg) / max(np.sum(occupancy), 1)),
        "direction_error_deg": {
            "sample_count": int(len(error)),
            "q50": float(np.quantile(error, 0.50)) if len(error) else float("nan"),
            "q95": float(np.quantile(error, 0.95)) if len(error) else float("nan"),
        },
    }
    if parting_mask is not None:
        region = parting_mask & seg
        occlusion = int(np.sum(occupancy & region))
        region_count = int(np.sum(region))
        metrics["parting_region"] = {
            "pixel_count": region_count,
            "hair_occlusion_ratio": float(occlusion / max(region_count, 1)),
            "scalp_visibility_ratio": float(1.0 - occlusion / max(region_count, 1)),
        }
    return metrics


def count_surface_penetration(step7_report: dict[str, object]) -> dict[str, int]:
    """沿用 Step 7 对实际替换选区的头模穿模精确结果。"""

    penetration = step7_report["metrics"]["penetration_points"]
    return {
        "selected_v30_baseline": int(penetration["selected_v30_baseline"]),
        "step7_coupled_layer": int(penetration["coupled_layer"]),
        "added": int(penetration["added"]),
    }


def compose_multiview_comparison(
    output_dir: Path,
    images: dict[str, np.ndarray],
    rasters: dict[str, dict[str, dict[str, object]]],
    parting_mask: np.ndarray,
) -> dict[str, str]:
    """输出三模型四视图总表和正面发缝局部放大图。"""

    colors = {"v30": (255, 170, 40), "v4_groom": (40, 170, 255), "step7": (60, 240, 80)}
    rows = []
    individual = {}
    for view in VIEWS:
        panels = []
        for model in MODELS:
            base = images[view].copy()
            occupancy = np.asarray(rasters[model][view]["occupancy"], dtype=bool)
            overlay = base.copy()
            overlay[occupancy] = colors[model]
            panel = cv2.addWeighted(base, 0.58, overlay, 0.42, 0.0)
            label = f"{model} / {view}"
            cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
            if model == "step7":
                path = output_dir / f"step7_{view}_overlay.png"
                cv2.imwrite(str(path), panel)
                individual[view] = str(path)
        rows.append(np.concatenate(panels, axis=1))
    grid = output_dir / "v30_v4_step7_multiview_comparison.png"
    cv2.imwrite(str(grid), np.concatenate(rows, axis=0))
    ys, xs = np.where(parting_mask)
    padding = 24
    x0, x1 = max(int(xs.min()) - padding, 0), min(int(xs.max()) + padding + 1, 512)
    y0, y1 = max(int(ys.min()) - padding, 0), min(int(ys.max()) + padding + 1, 512)
    crops = []
    for model in MODELS:
        base = images["front"].copy()
        occupancy = np.asarray(rasters[model]["front"]["occupancy"], dtype=bool)
        overlay = base.copy()
        overlay[occupancy] = colors[model]
        panel = cv2.addWeighted(base, 0.58, overlay, 0.42, 0.0)[y0:y1, x0:x1]
        panel = cv2.resize(panel, (360, 512), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, model, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        crops.append(panel)
    crop_path = output_dir / "front_parting_crop_raw_comparison.png"
    cv2.imwrite(str(crop_path), np.concatenate(crops, axis=1))
    return {"multiview_grid": str(grid), "front_parting_crop": str(crop_path), **{f"step7_{view}": path for view, path in individual.items()}}


def run_step8_render_gate(args: argparse.Namespace) -> dict[str, object]:
    """执行四视图软件可见性渲染、相对指标门禁和对比输出。"""

    config = VisibilityConfig()
    inputs = load_step8_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = {view: load_view_camera(args, view) for view in VIEWS}
    head_buffers = {
        view: build_head_depth_map(inputs["head"], cameras[view]["calib"], cameras[view]["toward_camera"], config.image_size)
        for view in VIEWS
    }
    rasters: dict[str, dict[str, dict[str, object]]] = {model: {} for model in MODELS}
    metrics: dict[str, dict[str, dict[str, object]]] = {model: {} for model in MODELS}
    for model in MODELS:
        for view in VIEWS:
            segments = collect_projected_segments(
                inputs["models"][model],
                cameras[view]["calib"],
                config.image_size,
                config.physical_sampling_m,
            )
            head_depth, origins, direction = head_buffers[view]
            raster = rasterize_hair_view(segments, head_depth, origins, direction, config)
            rasters[model][view] = raster
            metrics[model][view] = evaluate_model_view(
                segments,
                raster,
                inputs["seg"][view],
                inputs["strand_maps"][view],
                inputs["parting_mask"] if view == "front" else None,
            )
    penetration = count_surface_penetration(inputs["step7_report"])
    v30_front = metrics["v30"]["front"]
    new_front = metrics["step7"]["front"]
    visibility_delta = new_front["parting_region"]["scalp_visibility_ratio"] - v30_front["parting_region"]["scalp_visibility_ratio"]
    direction_delta = new_front["direction_error_deg"]["q50"] - v30_front["direction_error_deg"]["q50"]
    silhouette_deltas = {view: metrics["step7"][view]["silhouette_iou"] - metrics["v30"][view]["silhouette_iou"] for view in VIEWS}
    behind_deltas = {view: metrics["step7"][view]["behind_head_fraction"] - metrics["v30"][view]["behind_head_fraction"] for view in VIEWS}
    numeric_gates = {
        "front_parting_scalp_visibility_not_worse_than_v30": visibility_delta >= -1e-12,
        "front_direction_error_not_worse_over_2deg": direction_delta <= config.direction_error_allowance_deg,
        "all_view_silhouette_iou_not_worse_over_0p02": min(silhouette_deltas.values()) >= -config.silhouette_iou_allowance,
        "all_view_behind_head_fraction_not_worse_over_0p005": max(behind_deltas.values()) <= config.behind_head_fraction_allowance,
        "no_new_surface_penetration": penetration["added"] == 0,
    }
    outputs = compose_multiview_comparison(args.output_dir, inputs["images"], rasters, inputs["parting_mask"])
    report = {
        "image_id": args.image_id,
        "step": "step_08_render_gate",
        "inputs": {
            "v30": str(args.v30_hair.resolve()),
            "v4_groom": str(args.v4_groom_hair.resolve()),
            "step7": str(args.step7_hair.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
            "side_background_policy": "flux_redrawn view images; no Blender RGB used",
        },
        "config": config.__dict__,
        "metrics": {
            "models": metrics,
            "relative_to_v30": {
                "front_parting_scalp_visibility_delta": visibility_delta,
                "front_direction_error_q50_delta_deg": direction_delta,
                "silhouette_iou_delta": silhouette_deltas,
                "behind_head_fraction_delta": behind_deltas,
            },
            "surface_penetration": penetration,
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": outputs,
        "parent_step_report": str(args.step7_report.resolve()),
    }
    report_path = args.output_dir / "step_08_render_gate_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step8_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 8 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step7-report", type=Path, default=base / "connection_lb_parting/step_07_outer_coupling/step_07_outer_coupling_report.json")
    parser.add_argument("--v30-hair", type=Path, default=base / "pde_parting_hybrid_cap_v30/hair_multiview.ply")
    parser.add_argument("--v4-groom-hair", type=Path, default=base / "parting_groom_layer_smoke_v4_visible_alternating/hair_multiview.ply")
    parser.add_argument("--step7-hair", type=Path, default=base / "connection_lb_parting/step_07_outer_coupling/v30_merged_step7.ply")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--output-dir", type=Path, default=base / "connection_lb_parting/step_08_render_gate")
    return parser


def step8_main() -> None:
    """运行 Step 8 并打印门禁摘要。"""

    report = run_step8_render_gate(build_step8_arg_parser().parse_args())
    print(json.dumps({"passed_numeric_gates": report["passed_numeric_gates"], "numeric_gates": report["numeric_gates"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step8_main()
