#!/usr/bin/env python3
"""Step 10C-1：审计发缝可见遮挡采样点的保深度三维位移约束。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import read_ordered_strands
from scripts.recon_3d.couple_parting_outer_strands import resample_polyline
from scripts.vis.evaluate_parting_step8 import (
    VisibilityConfig,
    build_head_depth_map,
    load_view_camera,
    project_view,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step10c1_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 10B 已拒绝，并读取基线、真实遮挡 ID、发缝曲线和正面输入。"""

    parent = json.loads(args.step10b_report.read_text(encoding="utf-8"))
    if parent.get("human_review_status") != "rejected":
        raise RuntimeError("Step 10C-1 只允许在 Step 10B 被拒绝后运行")
    strands = read_ordered_strands(args.full_10k_baseline)
    if len(strands) != 10000:
        raise ValueError("Step 10C-1 基线必须严格包含 10000 根发丝")
    with np.load(args.visible_occluders, allow_pickle=False) as archive:
        occluder_ids = np.asarray(archive["strand_ids"], dtype=np.int64)
    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    parting = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or seg is None or parting is None:
        raise FileNotFoundError("缺少 raw_img.png、front seg 或 parting mask")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    seg = cv2.resize(seg, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    parting = cv2.resize(parting, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    head = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not head.has_vertices() or not head.has_triangles():
        raise ValueError("头模为空")
    return {
        "parent": parent,
        "strands": strands,
        "occluder_ids": occluder_ids,
        "curve": curve,
        "image": image,
        "seg": seg,
        "parting": parting,
        "region": seg & parting,
        "head": head,
    }


def build_parting_normal_model(curve_pixels: np.ndarray) -> dict[str, np.ndarray]:
    """把投影发缝折线表示为带一致局部切线和法向的线段集合。"""

    start = np.asarray(curve_pixels[:-1], dtype=np.float64)
    end = np.asarray(curve_pixels[1:], dtype=np.float64)
    vector = end - start
    length2 = np.sum(vector * vector, axis=1)
    valid = length2 > 1e-8
    start = start[valid]
    vector = vector[valid]
    length2 = length2[valid]
    tangent = vector / np.sqrt(length2)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return {
        "start": start,
        "vector": vector,
        "length2": length2,
        "tangent": tangent,
        "normal": normal,
    }


def _nearest_curve_coordinates(
    points: np.ndarray, model: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative = points[:, None, :] - model["start"][None, :, :]
    fraction = np.sum(relative * model["vector"][None, :, :], axis=2)
    fraction /= model["length2"][None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    closest_all = model["start"][None, :, :] + fraction[:, :, None] * model["vector"][None, :, :]
    distance2 = np.sum((points[:, None, :] - closest_all) ** 2, axis=2)
    segment = np.argmin(distance2, axis=1)
    rows = np.arange(len(points))
    closest = closest_all[rows, segment]
    normal = model["normal"][segment]
    signed = np.sum((points - closest) * normal, axis=1)
    return segment.astype(np.int32), closest, normal, signed


def lift_pixel_offsets_to_world(pixel_delta: np.ndarray, calib: np.ndarray, image_size: int) -> np.ndarray:
    """反解正交相机位移，并把 NDC 深度增量固定为零。"""

    ndc_delta = np.column_stack([
        2.0 * pixel_delta[:, 0] / max(image_size - 1, 1),
        2.0 * pixel_delta[:, 1] / max(image_size - 1, 1),
        np.zeros(len(pixel_delta), dtype=np.float64),
    ])
    return np.linalg.solve(np.asarray(calib[:3, :3], dtype=np.float64), ndc_delta.T).T


def extract_visibility_constraints(
    strands: list[np.ndarray],
    occluder_ids: np.ndarray,
    curve_model: dict[str, np.ndarray],
    calib: np.ndarray,
    head_depth: np.ndarray,
    ray_origins: np.ndarray,
    ray_direction: np.ndarray,
    region: np.ndarray,
    config: VisibilityConfig,
    clearance_px: int,
) -> dict[str, np.ndarray]:
    """提取真实可见遮挡中点，并生成分岸、保深度的目标位移。"""

    radius = int(config.occupancy_dilation_px)
    forbidden_radius = radius + int(clearance_px)
    forbidden = cv2.dilate(
        region.astype(np.uint8),
        np.ones((2 * forbidden_radius + 1, 2 * forbidden_radius + 1), np.uint8),
    ) > 0
    result: dict[str, list[np.ndarray | float | int]] = {
        "strand_id": [], "arclength_m": [], "point": [], "source_pixel": [],
        "target_pixel": [], "pixel_delta": [], "world_delta": [], "side": [],
        "curve_segment": [], "root_side_fallback": [],
    }
    for strand_id in occluder_ids:
        sampled, arclength = resample_polyline(strands[int(strand_id)], config.physical_sampling_m)
        midpoint = 0.5 * (sampled[:-1] + sampled[1:])
        midpoint_s = 0.5 * (arclength[:-1] + arclength[1:])
        midpoint_px, _ = project_view(midpoint, calib, config.image_size)
        px = np.rint(midpoint_px[:, 0]).astype(np.int32)
        py = np.rint(midpoint_px[:, 1]).astype(np.int32)
        inside = (px >= 0) & (px < config.image_size) & (py >= 0) & (py < config.image_size)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        hair_distance = np.sum(
            (midpoint[ids] - ray_origins[py[ids], px[ids]]) * ray_direction[None, :], axis=1
        )
        visible_ids = ids[
            hair_distance <= head_depth[py[ids], px[ids]] + config.head_visibility_tolerance_m
        ]
        if not len(visible_ids):
            continue
        footprint_hit = np.zeros(len(visible_ids), dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                qx = np.clip(px[visible_ids] + dx, 0, config.image_size - 1)
                qy = np.clip(py[visible_ids] + dy, 0, config.image_size - 1)
                footprint_hit |= region[qy, qx]
        constrained = visible_ids[footprint_hit]
        if not len(constrained):
            continue
        source = midpoint_px[constrained]
        segment, closest, normal, signed = _nearest_curve_coordinates(source, curve_model)
        root_px, _ = project_view(sampled[:1], calib, config.image_size)
        _, root_closest, root_normal, root_signed = _nearest_curve_coordinates(root_px, curve_model)
        fallback = float(root_signed[0])
        strand_side = -1.0 if fallback < 0.0 else 1.0
        side = np.full(len(signed), strand_side, dtype=np.float64)
        target = np.empty_like(source)
        for local_id in range(len(source)):
            chosen = None
            for distance in np.arange(max(abs(float(signed[local_id])), 0.0), 96.5, 0.5):
                candidate = closest[local_id] + side[local_id] * distance * normal[local_id]
                cx, cy = np.rint(candidate).astype(np.int32)
                if not (0 <= cx < config.image_size and 0 <= cy < config.image_size):
                    continue
                if not forbidden[cy, cx]:
                    chosen = candidate
                    break
            if chosen is None:
                raise RuntimeError(f"strand {strand_id} 的约束无法在 96 px 内离开发缝安全区")
            target[local_id] = chosen
        pixel_delta = target - source
        world_delta = lift_pixel_offsets_to_world(pixel_delta, calib, config.image_size)
        for local_id, sample_id in enumerate(constrained):
            result["strand_id"].append(int(strand_id))
            result["arclength_m"].append(float(midpoint_s[sample_id]))
            result["point"].append(midpoint[sample_id])
            result["source_pixel"].append(source[local_id])
            result["target_pixel"].append(target[local_id])
            result["pixel_delta"].append(pixel_delta[local_id])
            result["world_delta"].append(world_delta[local_id])
            result["side"].append(int(side[local_id]))
            result["curve_segment"].append(int(segment[local_id]))
            result["root_side_fallback"].append(int(strand_side))
    dtypes = {
        "strand_id": np.int32, "arclength_m": np.float32, "point": np.float32,
        "source_pixel": np.float32, "target_pixel": np.float32, "pixel_delta": np.float32,
        "world_delta": np.float32, "side": np.int8, "curve_segment": np.int32,
        "root_side_fallback": np.int8,
    }
    return {key: np.asarray(values, dtype=dtypes[key]) for key, values in result.items()}


def render_visibility_constraint_audit(
    output_dir: Path,
    image: np.ndarray,
    curve_pixels: np.ndarray,
    constraints: dict[str, np.ndarray],
    maximum_arrows: int,
) -> dict[str, str]:
    """在 raw_img.png 全图和发缝裁剪图上显示源点、目标点与位移箭头。"""

    source = np.asarray(constraints["source_pixel"])
    target = np.asarray(constraints["target_pixel"])
    side = np.asarray(constraints["side"])
    panels = []
    source_panel = image.copy()
    cv2.polylines(source_panel, [np.rint(curve_pixels).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 2, cv2.LINE_AA)
    for point in np.rint(source).astype(np.int32):
        cv2.circle(source_panel, tuple(point), 1, (40, 80, 255), -1, cv2.LINE_AA)
    cv2.putText(source_panel, "visible occlusion samples", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(source_panel)
    target_panel = image.copy()
    order = np.linspace(0, max(len(source) - 1, 0), min(maximum_arrows, len(source)), dtype=np.int64)
    for index in order:
        color = (255, 170, 30) if int(side[index]) < 0 else (30, 210, 255)
        a = tuple(np.rint(source[index]).astype(np.int32))
        b = tuple(np.rint(target[index]).astype(np.int32))
        cv2.arrowedLine(target_panel, a, b, color, 1, cv2.LINE_AA, tipLength=0.25)
    for point in np.rint(target).astype(np.int32):
        cv2.circle(target_panel, tuple(point), 1, (40, 230, 80), -1, cv2.LINE_AA)
    cv2.putText(target_panel, "bank targets / depth fixed", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(target_panel)
    full = np.concatenate(panels, axis=1)
    full_path = output_dir / "visibility_constraints_on_raw_img.png"
    cv2.imwrite(str(full_path), full)
    all_points = np.vstack([source, target, curve_pixels])
    x0 = max(int(np.floor(np.min(all_points[:, 0]))) - 18, 0)
    x1 = min(int(np.ceil(np.max(all_points[:, 0]))) + 19, image.shape[1])
    y0 = max(int(np.floor(np.min(all_points[:, 1]))) - 18, 0)
    y1 = min(int(np.ceil(np.max(all_points[:, 1]))) + 19, image.shape[0])
    crops = [cv2.resize(panel[y0:y1, x0:x1], (480, 512), interpolation=cv2.INTER_NEAREST) for panel in panels]
    crop_path = output_dir / "visibility_constraints_parting_crop_raw_img.png"
    cv2.imwrite(str(crop_path), np.concatenate(crops, axis=1))
    return {"full_raw_preview": str(full_path), "parting_crop_raw_preview": str(crop_path)}


def run_step10c1_constraint_audit(args: argparse.Namespace) -> dict[str, object]:
    """构建 Step 10C 的三维可见性约束场，但不改变任何发丝。"""

    inputs = load_step10c1_inputs(args)
    config = VisibilityConfig()
    camera = load_view_camera(args, "front")
    curve_pixels, _ = project_view(inputs["curve"], camera["calib"], config.image_size)
    curve_model = build_parting_normal_model(curve_pixels)
    head_depth, origins, direction = build_head_depth_map(
        inputs["head"], camera["calib"], camera["toward_camera"], config.image_size
    )
    constraints = extract_visibility_constraints(
        inputs["strands"], inputs["occluder_ids"], curve_model, camera["calib"],
        head_depth, origins, direction, inputs["region"], config, args.clearance_px,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_ids, counts = np.unique(constraints["strand_id"], return_counts=True)
    offsets = np.zeros(len(unique_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    data_path = args.output_dir / "visibility_constraints_3d.npz"
    np.savez_compressed(
        data_path, **constraints, constrained_strand_ids=unique_ids.astype(np.int32),
        strand_constraint_offsets=offsets, curve_pixels=curve_pixels.astype(np.float32),
    )
    projected_target, _ = project_view(
        constraints["point"].astype(np.float64) + constraints["world_delta"].astype(np.float64),
        camera["calib"], config.image_size,
    )
    reprojection_error = np.linalg.norm(projected_target - constraints["target_pixel"], axis=1)
    depth_delta = constraints["world_delta"].astype(np.float64) @ camera["calib"][2, :3]
    target_rounded = np.rint(constraints["target_pixel"]).astype(np.int32)
    forbidden_radius = config.occupancy_dilation_px + args.clearance_px
    forbidden = cv2.dilate(
        inputs["region"].astype(np.uint8),
        np.ones((2 * forbidden_radius + 1, 2 * forbidden_radius + 1), np.uint8),
    ) > 0
    target_violation = int(np.sum(forbidden[target_rounded[:, 1], target_rounded[:, 0]]))
    target_outside_seg = int(
        np.sum(~inputs["seg"][target_rounded[:, 1], target_rounded[:, 0]])
    )
    mixed_side_strands = int(
        sum(
            len(np.unique(constraints["side"][constraints["strand_id"] == strand_id])) > 1
            for strand_id in unique_ids
        )
    )
    pixel_norm = np.linalg.norm(constraints["pixel_delta"], axis=1)
    world_norm = np.linalg.norm(constraints["world_delta"], axis=1)
    coverage = len(unique_ids) / max(len(inputs["occluder_ids"]), 1)
    numeric_gates = {
        "occluder_constraint_coverage_ge_95pct": coverage >= 0.95,
        "target_outside_forbidden_corridor": target_violation == 0,
        "reprojection_error_q95_le_0p05px": float(np.quantile(reprojection_error, 0.95)) <= 0.05,
        "ndc_depth_delta_max_le_1e_8": float(np.max(np.abs(depth_delta))) <= 1e-8,
        "both_banks_present": set(np.unique(constraints["side"]).tolist()) == {-1, 1},
        "one_bank_per_strand": mixed_side_strands == 0,
        "all_targets_inside_front_seg": target_outside_seg == 0,
        "no_geometry_output": True,
    }
    outputs = render_visibility_constraint_audit(
        args.output_dir, inputs["image"], curve_pixels, constraints, args.maximum_arrows,
    )
    outputs["constraint_data"] = str(data_path)
    report = {
        "image_id": args.image_id,
        "step": "step_10c1_visibility_constraint_field_audit",
        "purpose": "为三维发束优化构造分岸、保相机深度的可见遮挡位移约束；不修改几何",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "visible_occluders": str(args.visible_occluders.resolve()),
            "curve": str(args.atlas_data.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
        },
        "config": {
            **config.__dict__, "clearance_px": args.clearance_px,
            "forbidden_radius_px": forbidden_radius,
            "world_lift": "solve camera linear transform with zero NDC depth delta",
        },
        "metrics": {
            "input_occluder_count": int(len(inputs["occluder_ids"])),
            "constrained_strand_count": int(len(unique_ids)),
            "occluder_constraint_coverage": float(coverage),
            "constraint_count": int(len(constraints["strand_id"])),
            "constraint_count_per_strand": {
                "q50": float(np.quantile(counts, 0.50)), "q95": float(np.quantile(counts, 0.95)), "max": int(np.max(counts)),
            },
            "side_counts": {str(side): int(np.sum(constraints["side"] == side)) for side in (-1, 1)},
            "pixel_displacement": {
                "q50": float(np.quantile(pixel_norm, 0.50)), "q95": float(np.quantile(pixel_norm, 0.95)), "max": float(np.max(pixel_norm)),
            },
            "world_displacement_m": {
                "q50": float(np.quantile(world_norm, 0.50)), "q95": float(np.quantile(world_norm, 0.95)), "max": float(np.max(world_norm)),
            },
            "reprojection_error_px": {"q95": float(np.quantile(reprojection_error, 0.95)), "max": float(np.max(reprojection_error))},
            "ndc_depth_delta_abs_max": float(np.max(np.abs(depth_delta))),
            "target_forbidden_violation_count": target_violation,
            "target_outside_seg_count": target_outside_seg,
            "mixed_side_strand_count": mixed_side_strands,
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "geometry_modified": False,
        "outputs": outputs,
        "parent_step_report": str(args.step10b_report.resolve()),
    }
    report_path = args.output_dir / "step_10c1_visibility_constraint_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step10c1_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 10C-1 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    connection = base / "connection_lb_parting"
    step10 = connection / "step_10_occlusion_density"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step10b-report", type=Path, default=step10 / "step_10b_occlusion_first_candidate/step_10b_occlusion_first_report.json")
    parser.add_argument("--visible-occluders", type=Path, default=step10 / "step_10a_visible_occluders/10k_baseline_visible_occluders.npz")
    parser.add_argument("--full-10k-baseline", type=Path, default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply")
    parser.add_argument("--atlas-data", type=Path, default=connection / "step_02_cut_atlas/cut_atlas_data.npz")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--clearance-px", type=int, default=2)
    parser.add_argument("--maximum-arrows", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=step10 / "step_10c_visibility_constrained/step_10c1_constraint_field")
    return parser


def step10c1_main() -> None:
    """运行 Step 10C-1 并打印约束场摘要。"""

    report = run_step10c1_constraint_audit(build_step10c1_arg_parser().parse_args())
    print(json.dumps({
        "passed_numeric_gates": report["passed_numeric_gates"],
        "numeric_gates": report["numeric_gates"],
        "metrics": report["metrics"],
        "outputs": report["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step10c1_main()
