#!/usr/bin/env python3
"""Step 7：把 Step 6 根段与 v30 外层尾段做局部、可回退的 C2 耦合。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import (
    build_curve_lateral_frame,
    curve_coordinates,
    read_ordered_strands,
    write_ordered_strands,
)
from scripts.recon_3d.solve_parting_bank_only_field import (
    calibration_from_param,
    project_points_to_image,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


@dataclass(frozen=True)
class CouplingConfig:
    """Step 7 固定参数；长度单位为米。"""

    guide_root_influence_m: float = 0.035
    handoff_m: float = 0.060
    bridge_spacing_m: float = 0.001
    kinematics_radius_m: float = 0.005
    influence_dilation_px: int = 12
    maximum_handoff_angle_deg: float = 10.0
    penetration_tolerance_m: float = -1e-5


def load_step7_inputs(args: argparse.Namespace) -> dict[str, object]:
    """验证 Step 6，读取 v30、根段、曲线、头模与正面原图。"""

    step6_report = json.loads(args.step6_report.read_text(encoding="utf-8"))
    if not step6_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 6 数值门禁未通过")
    with np.load(args.root_traces, allow_pickle=False) as archive:
        root_points = np.asarray(archive["points"], dtype=np.float64)
        chart_id = np.asarray(archive["chart_id"], dtype=np.int8)
    if root_points.ndim != 3 or root_points.shape[2] != 3:
        raise ValueError("Step 6 points 形状无效")
    if len(root_points) != 256 or set(np.unique(chart_id)) != {-1, 1}:
        raise ValueError("Step 6 必须包含左右两岸各 128 根轨迹")
    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    parting_mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    if image is None or parting_mask is None or seg is None:
        raise FileNotFoundError("无法读取 raw_img、parting mask 或 front seg")
    if parting_mask.shape != image.shape[:2] or seg.shape != image.shape[:2]:
        raise ValueError("parting mask、seg 与 raw_img 尺寸不一致")
    calib = calibration_from_param(args.front_calib)
    depth_calib = calibration_from_param(args.front_depth_calib)
    calib[2] = depth_calib[2]
    head = trimesh.load(str(args.head_mesh), process=True)
    if isinstance(head, trimesh.Scene):
        head = trimesh.util.concatenate(tuple(head.geometry.values()))
    if not isinstance(head, trimesh.Trimesh):
        raise ValueError("头模不是三角网格")
    return {
        "step6_report": step6_report,
        "root_points": root_points,
        "chart_id": chart_id,
        "curve": curve,
        "guides": read_ordered_strands(args.v30_hair),
        "image": image,
        "parting_mask": parting_mask,
        "seg": seg,
        "calib": calib,
        "head": head,
    }


def effective_polyline(points: np.ndarray) -> np.ndarray:
    """删除连续重复采样点，并截掉 v30 的静止尾部。"""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("折线必须为 (N, 3)")
    if len(points) < 2:
        raise ValueError("折线点数不足")
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8])
    clean = points[keep]
    if len(clean) < 2:
        raise ValueError("折线没有有效长度")
    return clean


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """返回折线逐点累计弧长。"""

    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def resample_polyline(points: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """以近似等弧长间距重采样折线。"""

    points = effective_polyline(points)
    source_arc = arc_lengths(points)
    targets = np.arange(0.0, source_arc[-1], float(spacing_m))
    targets = np.unique(np.concatenate([targets, [source_arc[-1]]]))
    sampled = np.column_stack(
        [np.interp(targets, source_arc, points[:, axis]) for axis in range(3)]
    )
    return sampled, targets


def sample_kinematics(points: np.ndarray, target_m: float, radius_m: float) -> dict[str, np.ndarray | float | int]:
    """用局部二次拟合估计指定弧长处的位置、单位切向和曲率向量。"""

    points = effective_polyline(points)
    arc = arc_lengths(points)
    if arc[-1] <= target_m + radius_m:
        raise ValueError("发束短于交接弧长及运动学窗口")
    handoff_id = int(np.searchsorted(arc, target_m, side="left"))
    handoff_id = min(max(handoff_id, 1), len(points) - 3)
    center = float(arc[handoff_id])
    local = np.abs(arc - center) <= radius_m
    ids = np.flatnonzero(local)
    if len(ids) < 5:
        begin = max(0, handoff_id - 2)
        stop = min(len(points), handoff_id + 3)
        ids = np.arange(begin, stop)
    x = arc[ids] - center
    coefficients = np.stack([np.polyfit(x, points[ids, axis], 2) for axis in range(3)])
    position = points[handoff_id].copy()
    tangent_raw = coefficients[:, 1]
    tangent = tangent_raw / max(np.linalg.norm(tangent_raw), 1e-12)
    curvature = 2.0 * coefficients[:, 0]
    curvature -= np.dot(curvature, tangent) * tangent
    outgoing = points[handoff_id + 1] - points[handoff_id]
    outgoing /= max(np.linalg.norm(outgoing), 1e-12)
    return {
        "position": position,
        "tangent": tangent,
        "curvature": curvature,
        "outgoing": outgoing,
        "handoff_id": handoff_id,
        "handoff_arc_m": center,
    }


def _mask_hits(points: np.ndarray, mask: np.ndarray, calib: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    pixels = np.rint(project_points_to_image(points, calib, image_shape)).astype(np.int64)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < mask.shape[0])
    )
    hits = np.zeros(len(points), dtype=bool)
    hits[valid] = mask[pixels[valid, 1], pixels[valid, 0]] > 0
    return hits


def identify_affected_guides(
    guides: list[np.ndarray],
    curve: np.ndarray,
    lateral: np.ndarray,
    parting_mask: np.ndarray,
    seg: np.ndarray,
    calib: np.ndarray,
    image_shape: tuple[int, ...],
    config: CouplingConfig,
) -> dict[str, np.ndarray | list[np.ndarray] | list[dict[str, object] | None]]:
    """标记根部受发缝影响或前 60 mm 投影遮挡发缝的 v30 发束。"""

    kernel_size = 2 * int(config.influence_dilation_px) + 1
    dilated = cv2.dilate(parting_mask, np.ones((kernel_size, kernel_size), np.uint8))
    roots = np.stack([np.asarray(strand, dtype=np.float64)[0] for strand in guides])
    signed, curve_distance, curve_index = curve_coordinates(roots, curve, lateral)
    side = np.where(signed >= 0.0, 1, -1).astype(np.int8)
    clean_guides: list[np.ndarray] = []
    kinematics: list[dict[str, object] | None] = []
    length = np.zeros(len(guides), dtype=np.float64)
    root_projection = _mask_hits(roots, dilated, calib, image_shape)
    root_in_seg = _mask_hits(roots, seg, calib, image_shape)
    occludes = np.zeros(len(guides), dtype=bool)
    for guide_id, guide in enumerate(guides):
        try:
            clean = effective_polyline(guide)
            clean_guides.append(clean)
            length[guide_id] = arc_lengths(clean)[-1]
            if length[guide_id] > config.handoff_m + config.kinematics_radius_m:
                kinematics.append(sample_kinematics(clean, config.handoff_m, config.kinematics_radius_m))
                arc = arc_lengths(clean)
                prefix = clean[arc <= config.handoff_m]
                occludes[guide_id] = bool(np.any(_mask_hits(prefix, parting_mask, calib, image_shape)))
            else:
                kinematics.append(None)
        except ValueError:
            clean_guides.append(np.asarray(guide, dtype=np.float64))
            kinematics.append(None)
    root_curve_influence = curve_distance <= config.guide_root_influence_m
    affected = root_curve_influence | root_projection | occludes
    valid = affected & (length > config.handoff_m + config.kinematics_radius_m)
    return {
        "clean_guides": clean_guides,
        "kinematics": kinematics,
        "roots": roots,
        "signed": signed,
        "side": side,
        "curve_distance": curve_distance,
        "curve_index": curve_index,
        "length": length,
        "root_curve_influence": root_curve_influence,
        "root_projection_influence": root_projection,
        "root_in_seg": root_in_seg,
        "occludes_parting": occludes,
        "valid": valid,
    }


def build_match_cost(
    root_points: np.ndarray,
    root_curve_index: np.ndarray,
    candidates: np.ndarray,
    guide_data: dict[str, object],
    curve_count: int,
) -> np.ndarray:
    """联合发缝纵向位置、交接点距离与切向构造匹配代价。"""

    root_end = root_points[:, -1]
    root_tangent = root_points[:, -1] - root_points[:, -3]
    root_tangent /= np.maximum(np.linalg.norm(root_tangent, axis=1, keepdims=True), 1e-12)
    guide_curve_index = np.asarray(guide_data["curve_index"])[candidates]
    guide_kinematics = guide_data["kinematics"]
    endpoint = np.stack([guide_kinematics[int(index)]["position"] for index in candidates])
    tangent = np.stack([guide_kinematics[int(index)]["tangent"] for index in candidates])
    curve_cost = np.abs(root_curve_index[:, None] - guide_curve_index[None, :]) / max(curve_count - 1, 1)
    distance = np.linalg.norm(root_end[:, None, :] - endpoint[None, :, :], axis=2)
    cosine = np.clip(root_tangent @ tangent.T, -1.0, 1.0)
    angle = np.arccos(cosine)
    root_distance = np.asarray(guide_data["curve_distance"])[candidates]
    return 4.0 * curve_cost + distance / 0.030 + 0.5 * angle / (0.5 * np.pi) + 0.25 * root_distance[None, :] / 0.035


def match_guides_global(
    root_points: np.ndarray,
    root_side: np.ndarray,
    curve: np.ndarray,
    lateral: np.ndarray,
    guide_data: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """按侧别用 Hungarian 求解唯一的全局最小代价匹配。"""

    _, _, root_curve_index = curve_coordinates(root_points[:, 0], curve, lateral)
    matched = np.full(len(root_points), -1, dtype=np.int64)
    costs = np.full(len(root_points), np.inf, dtype=np.float64)
    valid = np.asarray(guide_data["valid"], dtype=bool)
    guide_side = np.asarray(guide_data["side"], dtype=np.int8)
    for bank in (-1, 1):
        roots = np.flatnonzero(root_side == bank)
        candidates = np.flatnonzero(valid & (guide_side == bank))
        if len(candidates) < len(roots):
            raise RuntimeError(f"侧别 {bank} 只有 {len(candidates)} 根有效 v30 候选")
        matrix = build_match_cost(root_points[roots], root_curve_index[roots], candidates, guide_data, len(curve))
        rows, columns = linear_sum_assignment(matrix)
        matched[roots[rows]] = candidates[columns]
        costs[roots[rows]] = matrix[rows, columns]
    if np.any(matched < 0) or len(np.unique(matched)) != len(matched):
        raise RuntimeError("全局匹配没有产生 256 根唯一 v30 发束")
    return matched, costs


def solve_quintic_bridge(
    start: np.ndarray,
    start_tangent: np.ndarray,
    start_curvature: np.ndarray,
    end: np.ndarray,
    end_tangent: np.ndarray,
    end_curvature: np.ndarray,
    nominal_length_m: float,
    sample_count: int,
) -> np.ndarray:
    """求满足两端位置、一阶导和二阶导的五次 Hermite 桥。"""

    scale = float(nominal_length_m)
    coefficients = np.zeros((6, 3), dtype=np.float64)
    coefficients[0] = start
    coefficients[1] = scale * start_tangent
    coefficients[2] = 0.5 * scale * scale * start_curvature
    rhs = np.stack(
        [
            end - coefficients[0] - coefficients[1] - coefficients[2],
            scale * end_tangent - coefficients[1] - 2.0 * coefficients[2],
            scale * scale * end_curvature - 2.0 * coefficients[2],
        ]
    )
    matrix = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]])
    coefficients[3:] = np.linalg.solve(matrix, rhs)
    u = np.linspace(0.0, 1.0, int(sample_count) + 1)[1:]
    powers = np.column_stack([u**degree for degree in range(6)])
    return powers @ coefficients


def couple_strand(
    root_trace: np.ndarray,
    guide: np.ndarray,
    kinematics: dict[str, object],
    config: CouplingConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """保留 30 mm 根段，C2 桥接到 60 mm 处并接回原 v30 尾段。"""

    root_trace = effective_polyline(root_trace)
    root_arc = arc_lengths(root_trace)
    root_tangent = root_trace[-1] - root_trace[-3]
    root_tangent /= max(np.linalg.norm(root_tangent), 1e-12)
    root_step = max(float(np.mean(np.diff(root_arc)[-2:])), 1e-12)
    root_curvature = (root_trace[-1] - 2.0 * root_trace[-2] + root_trace[-3]) / (root_step * root_step)
    root_curvature -= np.dot(root_curvature, root_tangent) * root_tangent
    handoff_id = int(kinematics["handoff_id"])
    end = np.asarray(kinematics["position"], dtype=np.float64)
    end_tangent = np.asarray(kinematics["outgoing"], dtype=np.float64)
    end_curvature = np.asarray(kinematics["curvature"], dtype=np.float64)
    nominal = max(config.handoff_m - float(root_arc[-1]), config.bridge_spacing_m)
    count = max(4, int(round(nominal / config.bridge_spacing_m)))
    bridge = solve_quintic_bridge(
        root_trace[-1], root_tangent, root_curvature,
        end, end_tangent, end_curvature, nominal, count,
    )
    tail = guide[handoff_id + 1 :]
    coupled = np.concatenate([root_trace, bridge, tail], axis=0)
    incoming = bridge[-1] - bridge[-2]
    outgoing = tail[0] - bridge[-1]
    incoming /= max(np.linalg.norm(incoming), 1e-12)
    outgoing /= max(np.linalg.norm(outgoing), 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))))
    left_step = max(np.linalg.norm(bridge[-1] - bridge[-2]), 1e-12)
    right_step = max(np.linalg.norm(tail[0] - bridge[-1]), 1e-12)
    left_tangent = (bridge[-1] - bridge[-2]) / left_step
    right_tangent = (tail[0] - bridge[-1]) / right_step
    curvature_jump = float(np.linalg.norm(right_tangent - left_tangent) / max(0.5 * (left_step + right_step), 1e-12))
    return coupled, {
        "root_point_count": int(len(root_trace)),
        "bridge_point_count": int(len(bridge)),
        "tail_point_count": int(len(tail)),
        "handoff_original_index": handoff_id,
        "handoff_original_arc_m": float(kinematics["handoff_arc_m"]),
        "bridge_chord_m": float(np.linalg.norm(end - root_trace[-1])),
        "handoff_angle_deg": angle,
        "handoff_curvature_jump_inv_m": curvature_jump,
    }


def surface_signed_heights(head: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """按最近三角面法向估计点相对头模的有符号高度。"""

    closest, _, face_ids = trimesh.proximity.closest_point(head, points)
    normals = np.asarray(head.face_normals, dtype=np.float64)[face_ids]
    return np.sum((points - closest) * normals, axis=1)


def evaluate_coupling(
    original: list[np.ndarray],
    merged: list[np.ndarray],
    selected_ids: np.ndarray,
    local: list[np.ndarray],
    diagnostics: list[dict[str, float | int]],
    head: trimesh.Trimesh,
    config: CouplingConfig,
) -> dict[str, object]:
    """执行 Step 7 四项数值门禁，并用 v30 局部曲率标定阈值。"""

    selected = set(int(index) for index in selected_ids)
    unselected_max = 0.0
    for index, (before, after) in enumerate(zip(original, merged)):
        if index not in selected:
            if before.shape != after.shape:
                unselected_max = float("inf")
                break
            unselected_max = max(unselected_max, float(np.max(np.abs(before - after))))
    angles = np.asarray([item["handoff_angle_deg"] for item in diagnostics], dtype=np.float64)
    jumps = np.asarray([item["handoff_curvature_jump_inv_m"] for item in diagnostics], dtype=np.float64)
    baseline_jumps = []
    for guide_id, item in zip(selected_ids, diagnostics):
        sampled, _ = resample_polyline(original[int(guide_id)], config.bridge_spacing_m)
        center = min(int(round(float(item["handoff_original_arc_m"]) / config.bridge_spacing_m)), len(sampled) - 3)
        center = max(center, 2)
        tangent_before = sampled[center] - sampled[center - 1]
        tangent_after = sampled[center + 1] - sampled[center]
        tangent_before /= max(np.linalg.norm(tangent_before), 1e-12)
        tangent_after /= max(np.linalg.norm(tangent_after), 1e-12)
        baseline_jumps.append(float(np.linalg.norm(tangent_after - tangent_before) / config.bridge_spacing_m))
    calibrated_threshold = max(float(np.quantile(baseline_jumps, 0.95)) * 1.5, 5.0)
    original_points = np.concatenate([effective_polyline(original[int(index)]) for index in selected_ids])
    local_points = np.concatenate(local)
    baseline_penetration = int(np.sum(surface_signed_heights(head, original_points) < config.penetration_tolerance_m))
    merged_penetration = int(np.sum(surface_signed_heights(head, local_points) < config.penetration_tolerance_m))
    added_penetration = max(merged_penetration - baseline_penetration, 0)
    gates = {
        "unselected_point_difference_eq_0": unselected_max == 0.0,
        "handoff_angle_max_le_10deg": float(np.max(angles)) <= config.maximum_handoff_angle_deg,
        "curvature_jump_below_calibrated_threshold": float(np.max(jumps)) <= calibrated_threshold,
        "no_new_penetration": added_penetration == 0,
    }
    return {
        "passed_numeric_gates": bool(all(gates.values())),
        "numeric_gates": gates,
        "unselected_point_max_abs_difference_m": unselected_max,
        "handoff_angle_deg": {"q50": float(np.quantile(angles, 0.5)), "q95": float(np.quantile(angles, 0.95)), "max": float(np.max(angles))},
        "handoff_curvature_jump_inv_m": {"q50": float(np.quantile(jumps, 0.5)), "q95": float(np.quantile(jumps, 0.95)), "max": float(np.max(jumps)), "calibrated_threshold": calibrated_threshold},
        "penetration_points": {"selected_v30_baseline": baseline_penetration, "coupled_layer": merged_penetration, "added": added_penetration},
    }


def render_front_preview(
    output: Path,
    image: np.ndarray,
    calib: np.ndarray,
    before: list[np.ndarray],
    after: list[np.ndarray],
    side: np.ndarray,
    titles: tuple[str, str] = ("v30 selected before", "Step 7 coupled"),
) -> None:
    """在 raw_img 上并排绘制替换前和耦合后选区，不使用 Blender 渲染。"""

    panels = []
    for title, strands in zip(titles, (before, after)):
        panel = image.copy()
        overlay = panel.copy()
        for strand, bank in zip(strands, side):
            pixels = np.rint(project_points_to_image(strand, calib, image.shape)).astype(np.int32)
            color = (255, 150, 30) if int(bank) < 0 else (30, 210, 255)
            cv2.polylines(overlay, [pixels.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0.0)
        cv2.putText(panel, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(panels, axis=1))


def run_step7(args: argparse.Namespace) -> dict[str, object]:
    """执行候选标记、全局匹配、C2 耦合、合并与门禁评估。"""

    config = CouplingConfig(
        guide_root_influence_m=args.guide_root_influence_mm / 1000.0,
        handoff_m=args.handoff_mm / 1000.0,
        bridge_spacing_m=args.bridge_spacing_mm / 1000.0,
        influence_dilation_px=args.influence_dilation_px,
    )
    inputs = load_step7_inputs(args)
    root_points = np.asarray(inputs["root_points"])
    root_side = np.asarray(inputs["chart_id"], dtype=np.int8)
    curve = np.asarray(inputs["curve"])
    lateral = build_curve_lateral_frame(curve, root_points[:, 0], root_side)
    guide_data = identify_affected_guides(
        inputs["guides"], curve, lateral, inputs["parting_mask"], inputs["seg"],
        inputs["calib"], inputs["image"].shape, config,
    )
    matched_ids, match_cost = match_guides_global(root_points, root_side, curve, lateral, guide_data)
    local = []
    diagnostics = []
    clean_guides = guide_data["clean_guides"]
    kinematics = guide_data["kinematics"]
    for trace, guide_id in zip(root_points, matched_ids):
        strand, item = couple_strand(trace, clean_guides[int(guide_id)], kinematics[int(guide_id)], config)
        local.append(strand)
        diagnostics.append(item)
    original = inputs["guides"]
    merged = list(original)
    for guide_id, strand in zip(matched_ids, local):
        merged[int(guide_id)] = strand
    metrics = evaluate_coupling(original, merged, matched_ids, local, diagnostics, inputs["head"], config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_output = args.output_dir / "v30_before.ply"
    shutil.copy2(args.v30_hair, before_output)
    local_output = args.output_dir / "local_replacement_layer.ply"
    merged_output = args.output_dir / "v30_merged_step7.ply"
    write_ordered_strands(local, local_output)
    write_ordered_strands(merged, merged_output)
    selected_before = [original[int(index)] for index in matched_ids]
    preview = args.output_dir / "selected_before_after_on_raw_img.png"
    render_front_preview(preview, inputs["image"], inputs["calib"], selected_before, local, root_side)
    valid = np.asarray(guide_data["valid"], dtype=bool)
    guide_side = np.asarray(guide_data["side"], dtype=np.int8)
    report = {
        "image_id": args.image_id,
        "step": "step_07_outer_coupling",
        "purpose": "仅替换发缝影响区 v30 发束，以 C2 五次桥连接 Step 6 根段和原 v30 尾段",
        "parents": {"step6_report": str(args.step6_report), "v30_baseline": str(args.v30_hair), "continued_by_user": True},
        "config": {
            "guide_root_influence_mm": args.guide_root_influence_mm,
            "handoff_mm": args.handoff_mm,
            "bridge_spacing_mm": args.bridge_spacing_mm,
            "influence_dilation_px": args.influence_dilation_px,
            "coupling": "quintic Hermite position+tangent+curvature",
            "matching": "same-bank Hungarian global unique assignment",
        },
        "selection": {
            "v30_total": len(original),
            "selected": len(matched_ids),
            "selected_side_counts": {str(bank): int(np.sum(root_side == bank)) for bank in (-1, 1)},
            "valid_candidate_side_counts": {str(bank): int(np.sum(valid & (guide_side == bank))) for bank in (-1, 1)},
            "root_curve_influence_count": int(np.sum(guide_data["root_curve_influence"])),
            "root_projection_influence_count": int(np.sum(guide_data["root_projection_influence"])),
            "occludes_parting_count": int(np.sum(guide_data["occludes_parting"])),
            "selected_guide_ids": matched_ids.tolist(),
        },
        "matching": {
            "cost": {"q50": float(np.quantile(match_cost, 0.5)), "q95": float(np.quantile(match_cost, 0.95)), "max": float(np.max(match_cost))},
            "bridge_chord_m": {"q50": float(np.quantile([x["bridge_chord_m"] for x in diagnostics], 0.5)), "q95": float(np.quantile([x["bridge_chord_m"] for x in diagnostics], 0.95)), "max": float(np.max([x["bridge_chord_m"] for x in diagnostics]))},
        },
        "metrics": metrics,
        "numeric_gates": metrics["numeric_gates"],
        "passed_numeric_gates": metrics["passed_numeric_gates"],
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {"before_ply": str(before_output), "local_ply": str(local_output), "merged_ply": str(merged_output), "front_raw_preview": str(preview)},
    }
    np.savez_compressed(
        args.output_dir / "coupling_selection.npz",
        selected_guide_ids=matched_ids,
        root_trace_ids=np.arange(len(root_points), dtype=np.int32),
        side=root_side,
        match_cost=match_cost,
        handoff_angle_deg=np.asarray([x["handoff_angle_deg"] for x in diagnostics]),
        handoff_curvature_jump_inv_m=np.asarray([x["handoff_curvature_jump_inv_m"] for x in diagnostics]),
    )
    report_path = args.output_dir / "step_07_outer_coupling_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 7 命令行参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--step6-report", type=Path, default=base / "connection_lb_parting/step_06_root_traces/step_06_root_traces_report.json")
    parser.add_argument("--root-traces", type=Path, default=base / "connection_lb_parting/step_06_root_traces/root_traces.npz")
    parser.add_argument("--atlas-data", type=Path, default=base / "connection_lb_parting/step_02_cut_atlas/cut_atlas_data.npz")
    parser.add_argument("--v30-hair", type=Path, default=base / "pde_parting_hybrid_cap_v30/hair_multiview.ply")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--guide-root-influence-mm", type=float, default=35.0)
    parser.add_argument("--handoff-mm", type=float, default=60.0)
    parser.add_argument("--bridge-spacing-mm", type=float, default=1.0)
    parser.add_argument("--influence-dilation-px", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=base / "connection_lb_parting/step_07_outer_coupling")
    return parser


def main() -> None:
    """运行 Step 7 并打印最小摘要。"""

    report = run_step7(build_arg_parser().parse_args())
    print(json.dumps({"passed_numeric_gates": report["passed_numeric_gates"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
