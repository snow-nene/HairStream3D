#!/usr/bin/env python3
"""Step 10C-3：以主动遮挡集、碰撞投影和信赖域迭代求解发缝可见性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import read_ordered_strands, write_ordered_strands
from scripts.recon_3d.couple_parting_outer_strands import (
    effective_polyline,
    render_front_preview,
    surface_signed_heights,
)
from scripts.recon_3d.solve_visibility_constrained_strand_displacement import (
    _turn_angles,
    solve_strand_displacement,
)
from scripts.vis.audit_parting_visibility_constraints_3d import (
    build_parting_normal_model,
    extract_visibility_constraints,
)
from scripts.vis.evaluate_parting_step8 import (
    VisibilityConfig,
    build_head_depth_map,
    collect_projected_segments,
    evaluate_model_view,
    load_view_camera,
    project_view,
    rasterize_hair_view,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step10c3_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 10C-2 已拒绝，并加载基线、曲线、相机和门禁输入。"""

    parent = json.loads(args.step10c2_report.read_text(encoding="utf-8"))
    if parent.get("human_review_status") != "rejected":
        raise RuntimeError("Step 10C-3 只允许在 Step 10C-2 被人工拒绝后运行")
    strands = read_ordered_strands(args.full_10k_baseline)
    if len(strands) != 10000:
        raise ValueError("Step 10C-3 基线必须严格包含 10000 根发丝")
    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    parting = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    strand_map = cv2.imread(str(args.front_strand_map), cv2.IMREAD_COLOR)
    if image is None or seg is None or parting is None or strand_map is None:
        raise FileNotFoundError("缺少 raw_img、front seg、parting mask 或 strand_map")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    seg = cv2.resize(seg, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    parting = cv2.resize(parting, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    strand_map = cv2.resize(strand_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    head_tm = trimesh.load(str(args.head_mesh), process=True)
    if isinstance(head_tm, trimesh.Scene):
        head_tm = trimesh.util.concatenate(tuple(head_tm.geometry.values()))
    head_o3d = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not isinstance(head_tm, trimesh.Trimesh) or not head_o3d.has_triangles():
        raise ValueError("头模无效")
    config = VisibilityConfig()
    camera = load_view_camera(args, "front")
    head_depth, ray_origins, ray_direction = build_head_depth_map(
        head_o3d, camera["calib"], camera["toward_camera"], config.image_size
    )
    curve_pixels, _ = project_view(curve, camera["calib"], config.image_size)
    return {
        "parent": parent, "baseline": strands, "image": image, "seg": seg,
        "parting": parting, "region": seg & parting, "strand_map": strand_map,
        "head_tm": head_tm, "head_o3d": head_o3d, "config": config,
        "camera": camera, "head_depth": head_depth, "ray_origins": ray_origins,
        "ray_direction": ray_direction, "curve_pixels": curve_pixels,
        "curve_model": build_parting_normal_model(curve_pixels),
    }


def evaluate_front_state(
    strands: list[np.ndarray], inputs: dict[str, object]
) -> dict[str, object]:
    """按 Step 8 的同一软件光栅化口径评估完整 10k 正面状态。"""

    config = inputs["config"]
    segments = collect_projected_segments(
        strands, inputs["camera"]["calib"], config.image_size, config.physical_sampling_m
    )
    raster = rasterize_hair_view(
        segments, inputs["head_depth"], inputs["ray_origins"],
        inputs["ray_direction"], config,
    )
    metrics = evaluate_model_view(
        segments, raster, inputs["seg"], inputs["strand_map"], inputs["parting"]
    )
    return {"segments": segments, "raster": raster, "metrics": metrics}


def _active_constraints(
    strands: list[np.ndarray], inputs: dict[str, object], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    return extract_visibility_constraints(
        strands, np.arange(len(strands), dtype=np.int32), inputs["curve_model"],
        inputs["camera"]["calib"], inputs["head_depth"], inputs["ray_origins"],
        inputs["ray_direction"], inputs["region"], inputs["config"], args.clearance_px,
    )


def _constraint_summary(constraints: dict[str, np.ndarray]) -> dict[str, float | int]:
    norm = np.linalg.norm(np.asarray(constraints["pixel_delta"]), axis=1)
    return {
        "strand_count": int(len(np.unique(constraints["strand_id"]))),
        "constraint_count": int(len(norm)),
        "pixel_distance_sum": float(np.sum(norm)),
        "pixel_distance_q50": float(np.quantile(norm, 0.50)) if len(norm) else 0.0,
        "pixel_distance_q95": float(np.quantile(norm, 0.95)) if len(norm) else 0.0,
    }


def solve_active_set_increment(
    strands: list[np.ndarray], constraints: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """对当前主动遮挡发丝求平滑增量，并逐发丝限制最大步长。"""

    updates: dict[int, np.ndarray] = {}
    deltas: dict[int, np.ndarray] = {}
    for strand_id in np.unique(constraints["strand_id"]):
        mask = constraints["strand_id"] == strand_id
        solved = solve_strand_displacement(
            strands[int(strand_id)], constraints["arclength_m"][mask].astype(np.float64),
            constraints["world_delta"][mask].astype(np.float64), args,
        )
        points = effective_polyline(strands[int(strand_id)])
        delta = np.asarray(solved["points"]) - points
        maximum = float(np.max(np.linalg.norm(delta, axis=1)))
        if maximum > args.trust_region_mm / 1000.0:
            delta *= (args.trust_region_mm / 1000.0) / maximum
        delta[0] = 0.0
        updates[int(strand_id)] = points
        deltas[int(strand_id)] = delta
    return updates, deltas


def project_strands_outside_head(
    trial: list[np.ndarray], strand_ids: np.ndarray, baseline: list[np.ndarray],
    head: trimesh.Trimesh, clearance_m: float,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """把试探点沿最近面法向投到头模外，同时严格恢复根点。"""

    projected_count = 0
    maximum_shift = 0.0
    for strand_id in strand_ids:
        index = int(strand_id)
        points = np.asarray(trial[index], dtype=np.float64).copy()
        closest, _, face_ids = trimesh.proximity.closest_point(head, points)
        normals = np.asarray(head.face_normals, dtype=np.float64)[face_ids]
        height = np.sum((points - closest) * normals, axis=1)
        bad = height < clearance_m
        bad[0] = False
        shift = np.maximum(clearance_m - height, 0.0)
        points[bad] += shift[bad, None] * normals[bad]
        projected_count += int(np.sum(bad))
        maximum_shift = max(maximum_shift, float(np.max(shift[bad])) if np.any(bad) else 0.0)
        points[0] = np.asarray(baseline[index])[0]
        trial[index] = points
    return trial, {"projected_point_count": projected_count, "maximum_projection_m": maximum_shift}


def _penetration_count(
    strands: list[np.ndarray], strand_ids: np.ndarray, head: trimesh.Trimesh
) -> int:
    if not len(strand_ids):
        return 0
    points = np.concatenate([effective_polyline(strands[int(index)]) for index in strand_ids])
    return int(np.sum(surface_signed_heights(head, points) < -1e-5))


def try_trust_region_update(
    current: list[np.ndarray], base_points: dict[int, np.ndarray],
    deltas: dict[int, np.ndarray], current_front: dict[str, object],
    current_constraints: dict[str, np.ndarray], modified_ids: set[int],
    inputs: dict[str, object], args: argparse.Namespace,
) -> tuple[list[np.ndarray] | None, dict[str, object] | None, dict[str, np.ndarray] | None, dict[str, object]]:
    """线搜索试探步，只接受可见率单调且没有几何门禁回退的候选。"""

    active_ids = np.asarray(sorted(deltas), dtype=np.int32)
    baseline_front = inputs["baseline_front"]["metrics"]
    current_visibility = current_front["metrics"]["parting_region"]["scalp_visibility_ratio"]
    current_score = _constraint_summary(current_constraints)["pixel_distance_sum"]
    attempts = []
    for scale in args.line_search_scales:
        trial = list(current)
        for strand_id in active_ids:
            index = int(strand_id)
            trial[index] = base_points[index] + float(scale) * deltas[index]
            trial[index][0] = np.asarray(inputs["baseline"][index])[0]
        trial, collision = project_strands_outside_head(
            trial, active_ids, inputs["baseline"], inputs["head_tm"], args.surface_clearance_mm / 1000.0,
        )
        all_modified = np.asarray(sorted(modified_ids | set(active_ids.tolist())), dtype=np.int32)
        penetration = _penetration_count(trial, all_modified, inputs["head_tm"])
        front = evaluate_front_state(trial, inputs)
        metrics = front["metrics"]
        visibility = metrics["parting_region"]["scalp_visibility_ratio"]
        silhouette_delta = metrics["silhouette_iou"] - baseline_front["silhouette_iou"]
        direction_delta = metrics["direction_error_deg"]["q50"] - baseline_front["direction_error_deg"]["q50"]
        gate = (
            penetration == 0
            and visibility + 1e-12 >= current_visibility
            and silhouette_delta >= -0.02
            and direction_delta <= 2.0
        )
        trial_constraints = None
        trial_score = None
        if gate:
            trial_constraints = _active_constraints(trial, inputs, args)
            trial_score = float(_constraint_summary(trial_constraints)["pixel_distance_sum"])
            gate = visibility > current_visibility + 1e-12 or trial_score < current_score - 1e-6
        attempts.append({
            "scale": float(scale), "visibility": float(visibility),
            "silhouette_delta": float(silhouette_delta), "direction_q50_delta_deg": float(direction_delta),
            "penetration_points": penetration, "constraint_score": trial_score,
            "accepted": bool(gate), **collision,
        })
        if gate:
            return trial, front, trial_constraints, {"attempts": attempts, "accepted_scale": float(scale)}
    return None, None, None, {"attempts": attempts, "accepted_scale": None}


def render_step10c3_diagnostics(
    output_dir: Path, inputs: dict[str, object], current: list[np.ndarray],
    final_front: dict[str, object], modified_ids: np.ndarray,
) -> dict[str, str]:
    """仅在 raw_img.png 上输出完整可见率和修改发丝前后对比。"""

    panels = []
    for name, state, color in (
        ("baseline", inputs["baseline_front"], (255, 170, 30)),
        ("Step 10C-3", final_front, (40, 230, 80)),
    ):
        base = inputs["image"].copy()
        occupancy = np.asarray(state["raster"]["occupancy"], dtype=bool)
        overlay = base.copy()
        overlay[occupancy & inputs["parting"] & inputs["seg"]] = color
        panel = cv2.addWeighted(base, 0.60, overlay, 0.40, 0.0)
        ratio = state["metrics"]["parting_region"]["scalp_visibility_ratio"]
        cv2.putText(panel, f"{name} scalp visible {100.0 * ratio:.2f}%", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    visibility_path = output_dir / "absolute_parting_visibility_on_raw_img.png"
    cv2.imwrite(str(visibility_path), np.concatenate(panels, axis=1))
    strands_path = None
    if len(modified_ids):
        strands_path = output_dir / "modified_strands_before_after_on_raw_img.png"
        render_front_preview(
            strands_path, inputs["image"], inputs["camera"]["calib"],
            [inputs["baseline"][int(index)] for index in modified_ids],
            [current[int(index)] for index in modified_ids],
            np.ones(len(modified_ids), dtype=np.int8),
            titles=("baseline active strands", "Step 10C-3 accepted state"),
        )
    return {
        "absolute_visibility_preview": str(visibility_path),
        "modified_strands_preview": str(strands_path) if strands_path else None,
    }


def run_step10c3_active_set(args: argparse.Namespace) -> dict[str, object]:
    """迭代主动集、写出隔离候选，并执行绝对门禁。"""

    inputs = load_step10c3_inputs(args)
    inputs["baseline_front"] = evaluate_front_state(inputs["baseline"], inputs)
    current = list(inputs["baseline"])
    current_front = inputs["baseline_front"]
    constraints = _active_constraints(current, inputs, args)
    modified_ids: set[int] = set()
    history = []
    for iteration in range(args.max_iterations):
        before = _constraint_summary(constraints)
        if before["constraint_count"] == 0:
            history.append({"iteration": iteration, "status": "no_active_constraints", "before": before})
            break
        base_points, deltas = solve_active_set_increment(current, constraints, args)
        trial, trial_front, trial_constraints, search = try_trust_region_update(
            current, base_points, deltas, current_front, constraints, modified_ids, inputs, args,
        )
        record = {"iteration": iteration, "before": before, "line_search": search}
        if trial is None:
            record["status"] = "rejected_all_scales"
            history.append(record)
            break
        current = trial
        current_front = trial_front
        constraints = trial_constraints
        modified_ids.update(deltas)
        record["status"] = "accepted"
        record["after"] = _constraint_summary(constraints)
        record["front_visibility"] = float(current_front["metrics"]["parting_region"]["scalp_visibility_ratio"])
        history.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modified = np.asarray(sorted(modified_ids), dtype=np.int32)
    candidate_path = args.output_dir / "full_10k_iterative_active_set.ply"
    local_path = args.output_dir / "local_modified_active_set.ply"
    write_ordered_strands(current, candidate_path)
    if len(modified):
        write_ordered_strands([current[int(index)] for index in modified], local_path)
    baseline_metrics = inputs["baseline_front"]["metrics"]
    final_metrics = current_front["metrics"]
    visibility = final_metrics["parting_region"]["scalp_visibility_ratio"]
    silhouette_delta = final_metrics["silhouette_iou"] - baseline_metrics["silhouette_iou"]
    direction_delta = final_metrics["direction_error_deg"]["q50"] - baseline_metrics["direction_error_deg"]["q50"]
    root_displacement = [
        np.linalg.norm(current[int(index)][0] - inputs["baseline"][int(index)][0]) for index in modified
    ]
    unselected = set(range(10000)) - modified_ids
    unselected_max = max(
        (float(np.max(np.abs(current[index] - inputs["baseline"][index]))) for index in unselected),
        default=0.0,
    )
    penetration = _penetration_count(current, modified, inputs["head_tm"])
    if len(modified):
        before_turn = _turn_angles([inputs["baseline"][int(index)] for index in modified])
        after_turn = _turn_angles([current[int(index)] for index in modified])
        turn_delta = float(np.quantile(after_turn, 0.95) - np.quantile(before_turn, 0.95))
    else:
        turn_delta = 0.0
    visibility_history = [
        inputs["baseline_front"]["metrics"]["parting_region"]["scalp_visibility_ratio"]
    ] + [row["front_visibility"] for row in history if row["status"] == "accepted"]
    gates = {
        "exactly_10000_strands": len(current) == 10000,
        "root_displacement_max_le_1um": max(root_displacement, default=0.0) <= 1e-6,
        "unselected_strands_unchanged": unselected_max == 0.0,
        "no_new_surface_penetration": penetration == 0,
        "front_visibility_monotonic": bool(np.all(np.diff(visibility_history) >= -1e-12)),
        "absolute_front_scalp_visibility_ge_68pct": visibility >= 0.68,
        "front_silhouette_iou_delta_ge_minus_0p02": silhouette_delta >= -0.02,
        "front_direction_q50_delta_le_2deg": direction_delta <= 2.0,
        "turn_angle_q95_increase_le_10deg": turn_delta <= 10.0,
    }
    outputs = render_step10c3_diagnostics(args.output_dir, inputs, current, current_front, modified)
    history_path = args.output_dir / "iteration_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs.update({
        "candidate_ply": str(candidate_path),
        "local_ply": str(local_path) if len(modified) else None,
        "iteration_history": str(history_path),
    })
    report = {
        "image_id": args.image_id,
        "step": "step_10c3_iterative_visibility_active_set",
        "purpose": "以动态遮挡主动集、碰撞投影、信赖域和单调线搜索修正完整 10k 发缝可见性",
        "inputs": {"full_10k_baseline": str(args.full_10k_baseline.resolve()), "front_image": str(args.front_image.resolve()), "front_background_policy": "raw_img.png only"},
        "config": {"max_iterations": args.max_iterations, "trust_region_mm": args.trust_region_mm, "line_search_scales": args.line_search_scales, "surface_clearance_mm": args.surface_clearance_mm, "clearance_px": args.clearance_px, "visibility_weight": args.visibility_weight, "bending_weight": args.bending_weight, "strain_weight": args.strain_weight, "tip_anchor_weight": args.tip_anchor_weight, "root_lock_mm": args.root_lock_mm},
        "selection": {"modified_strand_count": int(len(modified)), "unselected_count": int(10000 - len(modified))},
        "metrics": {"baseline_front": baseline_metrics, "candidate_front": final_metrics, "front_visibility_delta": float(visibility - baseline_metrics["parting_region"]["scalp_visibility_ratio"]), "silhouette_iou_delta": float(silhouette_delta), "direction_q50_delta_deg": float(direction_delta), "root_displacement_m_max": max(root_displacement, default=0.0), "unselected_max_abs_delta_m": unselected_max, "candidate_penetration_points": penetration, "turn_angle_q95_delta_deg": turn_delta, "accepted_iteration_count": int(sum(row["status"] == "accepted" for row in history)), "final_active_constraints": _constraint_summary(constraints), "visibility_history": visibility_history},
        "numeric_gates": gates,
        "passed_numeric_gates": bool(all(gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": outputs,
        "parent_step_report": str(args.step10c2_report.resolve()),
    }
    report_path = args.output_dir / "step_10c3_iterative_active_set_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step10c3_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 10C-3 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    connection = base / "connection_lb_parting"
    step10c = connection / "step_10_occlusion_density/step_10c_visibility_constrained"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step10c2-report", type=Path, default=step10c / "step_10c2_sparse_displacement/step_10c2_sparse_displacement_report.json")
    parser.add_argument("--full-10k-baseline", type=Path, default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply")
    parser.add_argument("--atlas-data", type=Path, default=connection / "step_02_cut_atlas/cut_atlas_data.npz")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--front-strand-map", type=Path, default=DEFAULT_ROOT / "maps/strand_map/front.png")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--trust-region-mm", type=float, default=2.0)
    parser.add_argument("--line-search-scales", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--surface-clearance-mm", type=float, default=0.05)
    parser.add_argument("--clearance-px", type=int, default=2)
    parser.add_argument("--visibility-weight", type=float, default=500.0)
    parser.add_argument("--bending-weight", type=float, default=10.0)
    parser.add_argument("--strain-weight", type=float, default=1.0)
    parser.add_argument("--tip-anchor-weight", type=float, default=2.0)
    parser.add_argument("--root-lock-mm", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=step10c / "step_10c3_iterative_active_set")
    return parser


def step10c3_main() -> None:
    """运行 Step 10C-3 并打印本轮检查摘要。"""

    report = run_step10c3_active_set(build_step10c3_arg_parser().parse_args())
    print(json.dumps({"passed_numeric_gates": report["passed_numeric_gates"], "numeric_gates": report["numeric_gates"], "metrics": report["metrics"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step10c3_main()
