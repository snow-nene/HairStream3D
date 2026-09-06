#!/usr/bin/env python3
"""Step 10C-2：求解可见性约束下的逐发丝稀疏三维位移。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import read_ordered_strands, write_ordered_strands
from scripts.recon_3d.couple_parting_outer_strands import (
    arc_lengths,
    effective_polyline,
    render_front_preview,
    surface_signed_heights,
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


def load_step10c2_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 10C-1 已接受，并读取基线、三维约束及正面门禁输入。"""

    parent = json.loads(args.step10c1_report.read_text(encoding="utf-8"))
    if parent.get("human_review_status") != "accepted":
        raise RuntimeError("Step 10C-1 尚未被人工标记为 accepted")
    if not parent.get("passed_numeric_gates", False):
        raise RuntimeError("Step 10C-1 数值门禁未通过")
    strands = read_ordered_strands(args.full_10k_baseline)
    if len(strands) != 10000:
        raise ValueError("Step 10C-2 基线必须严格包含 10000 根发丝")
    with np.load(args.constraints, allow_pickle=False) as archive:
        constraints = {key: np.asarray(archive[key]) for key in archive.files}
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
    return {
        "parent": parent,
        "strands": strands,
        "constraints": constraints,
        "image": image,
        "seg": seg,
        "parting": parting,
        "strand_map": strand_map,
        "head_tm": head_tm,
        "head_o3d": head_o3d,
    }


def build_interpolation_matrix(
    vertex_arclength: np.ndarray, constraint_arclength: np.ndarray
) -> sparse.csr_matrix:
    """构造把折线顶点位移线性插值到约束弧长的稀疏矩阵。"""

    n = len(vertex_arclength)
    target = np.clip(constraint_arclength, vertex_arclength[0], vertex_arclength[-1])
    right = np.searchsorted(vertex_arclength, target, side="right")
    right = np.clip(right, 1, n - 1)
    left = right - 1
    span = np.maximum(vertex_arclength[right] - vertex_arclength[left], 1e-12)
    fraction = (target - vertex_arclength[left]) / span
    rows = np.repeat(np.arange(len(target)), 2)
    cols = np.column_stack([left, right]).reshape(-1)
    values = np.column_stack([1.0 - fraction, fraction]).reshape(-1)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(target), n))


def solve_strand_displacement(
    points: np.ndarray,
    constraint_arclength: np.ndarray,
    target_displacement: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float | int]:
    """固定根段，用可见性、弯曲、应变和尾端软锚项求解三维位移。"""

    points = effective_polyline(points)
    arclength = arc_lengths(points)
    n = len(points)
    h = build_interpolation_matrix(arclength, constraint_arclength)
    d1 = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csr")
    d2 = sparse.diags([np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format="csr")
    tip = sparse.csr_matrix(([1.0], ([0], [n - 1])), shape=(1, n))
    matrix = (
        args.visibility_weight * (h.T @ h)
        + args.strain_weight * (d1.T @ d1)
        + args.bending_weight * (d2.T @ d2)
        + args.tip_anchor_weight * (tip.T @ tip)
        + sparse.eye(n, format="csr") * 1e-9
    ).tocsc()
    rhs = args.visibility_weight * (h.T @ target_displacement)
    locked = arclength <= args.root_lock_mm / 1000.0 + 1e-12
    free = np.flatnonzero(~locked)
    displacement = np.zeros((n, 3), dtype=np.float64)
    if len(free):
        reduced = matrix[free][:, free]
        for axis in range(3):
            displacement[free, axis] = spsolve(reduced, np.asarray(rhs)[free, axis])
    residual = h @ displacement - target_displacement
    return {
        "points": points + displacement,
        "displacement": displacement,
        "residual": residual,
        "constraint_prediction": h @ displacement,
        "locked_count": int(np.sum(locked)),
        "condition_proxy": float(matrix.diagonal().max() / max(matrix.diagonal().min(), 1e-15)),
    }


def build_sparse_displacement_candidate(
    inputs: dict[str, object], args: argparse.Namespace
) -> dict[str, object]:
    """对 Step 10C-1 的全部发丝独立求解，并保持其余 10k 发丝逐点不变。"""

    constraints = inputs["constraints"]
    selected_ids = np.asarray(constraints["constrained_strand_ids"], dtype=np.int64)
    offsets = np.asarray(constraints["strand_constraint_offsets"], dtype=np.int64)
    merged = list(inputs["strands"])
    local = []
    residuals = []
    predictions = []
    displacements = []
    locked_counts = []
    condition_proxy = []
    for local_id, strand_id in enumerate(selected_ids):
        begin, end = int(offsets[local_id]), int(offsets[local_id + 1])
        solved = solve_strand_displacement(
            inputs["strands"][int(strand_id)],
            np.asarray(constraints["arclength_m"][begin:end], dtype=np.float64),
            np.asarray(constraints["world_delta"][begin:end], dtype=np.float64),
            args,
        )
        merged[int(strand_id)] = solved["points"]
        local.append(solved["points"])
        residuals.append(solved["residual"])
        predictions.append(solved["constraint_prediction"])
        displacements.append(solved["displacement"])
        locked_counts.append(solved["locked_count"])
        condition_proxy.append(solved["condition_proxy"])
    return {
        "merged": merged,
        "local": local,
        "selected_ids": selected_ids,
        "residual": np.concatenate(residuals),
        "prediction": np.concatenate(predictions),
        "displacements": displacements,
        "locked_counts": np.asarray(locked_counts, dtype=np.int32),
        "condition_proxy": np.asarray(condition_proxy, dtype=np.float64),
    }


def _turn_angles(strands: list[np.ndarray]) -> np.ndarray:
    parts = []
    for strand in strands:
        points = effective_polyline(strand)
        tangent = np.diff(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
        if len(tangent) > 1:
            parts.append(np.degrees(np.arccos(np.clip(np.sum(tangent[:-1] * tangent[1:], axis=1), -1.0, 1.0))))
    return np.concatenate(parts)


def evaluate_step10c2_candidate(
    inputs: dict[str, object], candidate: dict[str, object], args: argparse.Namespace
) -> dict[str, object]:
    """执行约束残差、根点、穿模、平滑度和正面绝对可见率门禁。"""

    original = inputs["strands"]
    merged = candidate["merged"]
    selected = set(candidate["selected_ids"].tolist())
    unselected_max = 0.0
    for strand_id, (before, after) in enumerate(zip(original, merged)):
        if strand_id in selected:
            continue
        if before.shape != after.shape:
            unselected_max = float("inf")
            break
        unselected_max = max(unselected_max, float(np.max(np.abs(before - after))))
    root_displacement = np.asarray([
        np.linalg.norm(candidate["local"][local_id][0] - original[int(strand_id)][0])
        for local_id, strand_id in enumerate(candidate["selected_ids"])
    ])
    before_points = np.concatenate([effective_polyline(original[int(index)]) for index in candidate["selected_ids"]])
    after_points = np.concatenate(candidate["local"])
    baseline_penetration = int(np.sum(surface_signed_heights(inputs["head_tm"], before_points) < -1e-5))
    candidate_penetration = int(np.sum(surface_signed_heights(inputs["head_tm"], after_points) < -1e-5))
    config = VisibilityConfig()
    camera = load_view_camera(args, "front")
    head_depth, origins, direction = build_head_depth_map(
        inputs["head_o3d"], camera["calib"], camera["toward_camera"], config.image_size
    )
    model_metrics = {}
    rasters = {}
    for name, strands in (("baseline", original), ("candidate", merged)):
        segments = collect_projected_segments(strands, camera["calib"], config.image_size, config.physical_sampling_m)
        raster = rasterize_hair_view(segments, head_depth, origins, direction, config)
        model_metrics[name] = evaluate_model_view(
            segments, raster, inputs["seg"], inputs["strand_map"], inputs["parting"]
        )
        rasters[name] = raster
    residual_norm = np.linalg.norm(candidate["residual"], axis=1)
    predicted_world = candidate["prediction"]
    linear = camera["calib"][:3, :3]
    predicted_ndc = predicted_world @ linear.T
    target_ndc = np.asarray(inputs["constraints"]["world_delta"]) @ linear.T
    residual_px = np.linalg.norm((predicted_ndc[:, :2] - target_ndc[:, :2]) * 0.5 * (config.image_size - 1), axis=1)
    displacement_norm = np.concatenate([np.linalg.norm(value, axis=1) for value in candidate["displacements"]])
    before_turn = _turn_angles([original[int(index)] for index in candidate["selected_ids"]])
    after_turn = _turn_angles(candidate["local"])
    visibility = model_metrics["candidate"]["parting_region"]["scalp_visibility_ratio"]
    silhouette_delta = model_metrics["candidate"]["silhouette_iou"] - model_metrics["baseline"]["silhouette_iou"]
    direction_delta = model_metrics["candidate"]["direction_error_deg"]["q50"] - model_metrics["baseline"]["direction_error_deg"]["q50"]
    numeric_gates = {
        "exactly_10000_strands": len(merged) == 10000,
        "all_1289_occluders_solved": len(candidate["selected_ids"]) == 1289,
        "unselected_strands_unchanged": unselected_max == 0.0,
        "root_displacement_max_le_1um": float(np.max(root_displacement)) <= 1e-6,
        "constraint_reprojection_residual_q95_le_1px": float(np.quantile(residual_px, 0.95)) <= 1.0,
        "absolute_front_scalp_visibility_ge_68pct": float(visibility) >= 0.68,
        "no_new_surface_penetration": candidate_penetration <= baseline_penetration,
        "front_silhouette_iou_delta_ge_minus_0p02": float(silhouette_delta) >= -0.02,
        "front_direction_q50_delta_le_2deg": float(direction_delta) <= 2.0,
        "turn_angle_q95_increase_le_10deg": float(np.quantile(after_turn, 0.95) - np.quantile(before_turn, 0.95)) <= 10.0,
    }
    return {
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "constraint_residual_m": {"q50": float(np.quantile(residual_norm, 0.50)), "q95": float(np.quantile(residual_norm, 0.95)), "max": float(np.max(residual_norm))},
        "constraint_reprojection_residual_px": {"q50": float(np.quantile(residual_px, 0.50)), "q95": float(np.quantile(residual_px, 0.95)), "max": float(np.max(residual_px))},
        "displacement_m": {"q50": float(np.quantile(displacement_norm, 0.50)), "q95": float(np.quantile(displacement_norm, 0.95)), "max": float(np.max(displacement_norm))},
        "root_displacement_m_max": float(np.max(root_displacement)),
        "turn_angle_deg": {
            "baseline_q95": float(np.quantile(before_turn, 0.95)),
            "candidate_q95": float(np.quantile(after_turn, 0.95)),
            "q95_delta": float(np.quantile(after_turn, 0.95) - np.quantile(before_turn, 0.95)),
        },
        "penetration_points": {"baseline_selected": baseline_penetration, "candidate_selected": candidate_penetration, "added": max(candidate_penetration - baseline_penetration, 0)},
        "front": {"models": model_metrics, "silhouette_iou_delta": float(silhouette_delta), "direction_q50_delta_deg": float(direction_delta)},
        "rasters": rasters,
    }


def render_step10c2_preview(
    output_dir: Path,
    inputs: dict[str, object],
    candidate: dict[str, object],
    evaluation: dict[str, object],
    calib: np.ndarray,
) -> dict[str, str]:
    """用 raw_img.png 输出选区几何和正面绝对发缝占据对比。"""

    constraints = inputs["constraints"]
    side_by_id = dict(zip(constraints["constrained_strand_ids"].tolist(), [0] * len(constraints["constrained_strand_ids"])))
    for strand_id in constraints["constrained_strand_ids"]:
        values = constraints["side"][constraints["strand_id"] == strand_id]
        side_by_id[int(strand_id)] = int(values[0])
    side = np.asarray([side_by_id[int(index)] for index in candidate["selected_ids"]], dtype=np.int8)
    strands_path = output_dir / "selected_before_after_on_raw_img.png"
    render_front_preview(
        strands_path, inputs["image"], calib,
        [inputs["strands"][int(index)] for index in candidate["selected_ids"]],
        candidate["local"], side,
        titles=("baseline visible occluders", "Step 10C-2 sparse displacement"),
    )
    panels = []
    for name, color in (("baseline", (255, 170, 30)), ("candidate", (40, 230, 80))):
        base = inputs["image"].copy()
        occupancy = np.asarray(evaluation["rasters"][name]["occupancy"], dtype=bool)
        overlay = base.copy()
        overlay[occupancy & inputs["parting"] & inputs["seg"]] = color
        panel = cv2.addWeighted(base, 0.60, overlay, 0.40, 0.0)
        ratio = evaluation["front"]["models"][name]["parting_region"]["scalp_visibility_ratio"]
        cv2.putText(panel, f"{name} scalp visible {100.0 * ratio:.2f}%", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    visibility_path = output_dir / "absolute_parting_visibility_on_raw_img.png"
    cv2.imwrite(str(visibility_path), np.concatenate(panels, axis=1))
    return {"selected_strands_preview": str(strands_path), "absolute_visibility_preview": str(visibility_path)}


def run_step10c2_sparse_solve(args: argparse.Namespace) -> dict[str, object]:
    """求解、保存并评估 Step 10C-2 的完整 10k 隔离候选。"""

    inputs = load_step10c2_inputs(args)
    candidate = build_sparse_displacement_candidate(inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "full_10k_visibility_constrained.ply"
    local_path = args.output_dir / "local_visibility_constrained_1289.ply"
    write_ordered_strands(candidate["merged"], candidate_path)
    write_ordered_strands(candidate["local"], local_path)
    evaluation = evaluate_step10c2_candidate(inputs, candidate, args)
    camera = load_view_camera(args, "front")
    outputs = render_step10c2_preview(args.output_dir, inputs, candidate, evaluation, camera["calib"])
    outputs.update({"candidate_ply": str(candidate_path), "local_ply": str(local_path)})
    metrics = {key: value for key, value in evaluation.items() if key != "rasters"}
    np.savez_compressed(
        args.output_dir / "step10c2_solution.npz",
        selected_ids=candidate["selected_ids"],
        constraint_residual=candidate["residual"].astype(np.float32),
        constraint_prediction=candidate["prediction"].astype(np.float32),
        locked_counts=candidate["locked_counts"],
        condition_proxy=candidate["condition_proxy"],
    )
    outputs["solution_data"] = str(args.output_dir / "step10c2_solution.npz")
    report = {
        "image_id": args.image_id,
        "step": "step_10c2_sparse_visibility_displacement",
        "purpose": "固定根段并以可见性、弯曲、应变和尾端软锚能量求解完整 10k 隔离候选",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "constraints": str(args.constraints.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
        },
        "config": {
            "visibility_weight": args.visibility_weight,
            "bending_weight": args.bending_weight,
            "strain_weight": args.strain_weight,
            "tip_anchor_weight": args.tip_anchor_weight,
            "root_lock_mm": args.root_lock_mm,
            "solver": "independent sparse SPD normal equations per strand and coordinate",
        },
        "selection": {"selected_occluder_count": int(len(candidate["selected_ids"])), "unselected_count": int(10000 - len(candidate["selected_ids"]))},
        "metrics": metrics,
        "numeric_gates": evaluation["numeric_gates"],
        "passed_numeric_gates": evaluation["passed_numeric_gates"],
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": outputs,
        "parent_step_report": str(args.step10c1_report.resolve()),
    }
    report_path = args.output_dir / "step_10c2_sparse_displacement_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step10c2_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 10C-2 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    step10 = base / "connection_lb_parting/step_10_occlusion_density"
    step10c = step10 / "step_10c_visibility_constrained"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step10c1-report", type=Path, default=step10c / "step_10c1_constraint_field/step_10c1_visibility_constraint_report.json")
    parser.add_argument("--constraints", type=Path, default=step10c / "step_10c1_constraint_field/visibility_constraints_3d.npz")
    parser.add_argument("--full-10k-baseline", type=Path, default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--front-strand-map", type=Path, default=DEFAULT_ROOT / "maps/strand_map/front.png")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--visibility-weight", type=float, default=100.0)
    parser.add_argument("--bending-weight", type=float, default=10.0)
    parser.add_argument("--strain-weight", type=float, default=1.0)
    parser.add_argument("--tip-anchor-weight", type=float, default=2.0)
    parser.add_argument("--root-lock-mm", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=step10c / "step_10c2_sparse_displacement")
    return parser


def step10c2_main() -> None:
    """运行 Step 10C-2 并打印绝对门禁摘要。"""

    report = run_step10c2_sparse_solve(build_step10c2_arg_parser().parse_args())
    print(json.dumps({
        "passed_numeric_gates": report["passed_numeric_gates"],
        "numeric_gates": report["numeric_gates"],
        "front_parting": report["metrics"]["front"]["models"]["candidate"]["parting_region"],
        "outputs": report["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step10c2_main()
