#!/usr/bin/env python3
"""Step 10B：按真实正面遮挡集合重建 Connection-LB 10k 隔离候选。"""

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
    CouplingConfig,
    arc_lengths,
    couple_strand,
    effective_polyline,
    render_front_preview,
    sample_kinematics,
    surface_signed_heights,
)
from scripts.recon_3d.integrate_parting_root_traces import (
    barycentric_coordinates,
    build_triangle_walk_cache,
    lift_root_traces,
    walk_surface_step,
)
from scripts.recon_3d.solve_parting_bank_only_field import recover_line_tangents
from scripts.recon_3d.validate_parting_connection_lb import build_vertex_frames
from scripts.vis.evaluate_parting_step8 import (
    VisibilityConfig,
    build_head_depth_map,
    collect_projected_segments,
    load_view_camera,
    rasterize_hair_view,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step10b_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 10A 已接受，并读取遮挡集合、双 atlas、线场和完整基线。"""

    parent = json.loads(args.step10a_report.read_text(encoding="utf-8"))
    if parent.get("human_review_status") != "accepted":
        raise RuntimeError("Step 10A 尚未被人工标记为 accepted")
    if parent.get("geometry_modified") is not False:
        raise RuntimeError("Step 10A 必须是只读审计")
    strands = read_ordered_strands(args.full_10k_baseline)
    if len(strands) != 10000:
        raise ValueError("Step 10B 基线必须严格包含 10000 根发丝")
    with np.load(args.visible_occluders, allow_pickle=False) as archive:
        occluders = {key: np.asarray(archive[key]) for key in archive.files}
    strand_ids = np.asarray(occluders["strand_ids"], dtype=np.int64)
    if len(strand_ids) != len(np.unique(strand_ids)) or np.any((strand_ids < 0) | (strand_ids >= 10000)):
        raise ValueError("Step 10A 遮挡发丝 ID 无效")
    meshes: dict[str, trimesh.Trimesh] = {}
    for name, path in (("left", args.left_atlas), ("right", args.right_atlas)):
        mesh = trimesh.load(str(path), process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"{name} atlas 无效")
        meshes[name] = mesh
    with np.load(args.observed_fields, allow_pickle=False) as archive:
        fields = {
            "left": np.asarray(archive["left_q"], dtype=np.complex128),
            "right": np.asarray(archive["right_q"], dtype=np.complex128),
        }
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    parting = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or seg is None or parting is None:
        raise FileNotFoundError("缺少 raw_img.png、front seg 或 parting mask")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    seg = cv2.resize(seg, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    parting = cv2.resize(parting, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    head_tm = trimesh.load(str(args.head_mesh), process=True)
    if isinstance(head_tm, trimesh.Scene):
        head_tm = trimesh.util.concatenate(tuple(head_tm.geometry.values()))
    head_o3d = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not isinstance(head_tm, trimesh.Trimesh) or not head_o3d.has_triangles():
        raise ValueError("头模无效")
    return {
        "parent": parent,
        "strands": strands,
        "occluders": occluders,
        "meshes": meshes,
        "fields": fields,
        "image": image,
        "seg": seg,
        "parting": parting,
        "head_tm": head_tm,
        "head_o3d": head_o3d,
    }


def prepare_chart_direction_field(mesh: trimesh.Trimesh, q: np.ndarray) -> dict[str, object]:
    """缓存 atlas 邻接和由二重角 Connection-LB 解恢复的分片切向场。"""

    cache = build_triangle_walk_cache(mesh)
    frames = build_vertex_frames(mesh)
    vertex = recover_line_tangents(q / np.maximum(np.abs(q), 1e-15), frames)
    face_tangents = vertex[cache["faces"]].copy()
    for local in (1, 2):
        flip = np.sum(face_tangents[:, 0] * face_tangents[:, local], axis=1) < 0.0
        face_tangents[flip, local] *= -1.0
    cache["face_tangents"] = face_tangents
    cache["field_magnitude"] = np.abs(q)
    return cache


def integrate_projected_root_trace(
    cache: dict[str, object],
    root_surface_point: np.ndarray,
    root_face_id: int,
    original_tangent: np.ndarray,
    step_length_m: float,
    total_length_m: float,
) -> dict[str, object]:
    """从原根在 atlas 的最近投影点积分，并优先保持原发丝的有向切线。"""

    face_id = int(root_face_id)
    triangle = cache["vertices"][cache["faces"][face_id]]
    barycentric = barycentric_coordinates(root_surface_point, triangle)
    barycentric = np.clip(barycentric, 0.0, None)
    barycentric /= np.sum(barycentric)
    normal = cache["face_normals"][face_id]
    reference = np.asarray(original_tangent, dtype=np.float64)
    reference -= np.dot(reference, normal) * normal
    if np.linalg.norm(reference) < 1e-10:
        reference = barycentric @ cache["face_tangents"][face_id]
    reference /= max(float(np.linalg.norm(reference)), 1e-15)

    def attempt(initial: np.ndarray) -> dict[str, object]:
        current_face = face_id
        current_barycentric = barycentric.copy()
        current_reference = initial.copy()
        points = [root_surface_point]
        faces = [current_face]
        weights = [current_barycentric.copy()]
        confidence = [float(current_barycentric @ cache["field_magnitude"][cache["faces"][current_face]])]
        crossings = 0
        status = "complete"
        for _ in range(int(round(total_length_m / step_length_m))):
            result = walk_surface_step(
                cache, cache["face_tangents"], cache["field_magnitude"],
                current_face, current_barycentric, current_reference, step_length_m,
            )
            if result["status"] != "ok":
                status = str(result["status"])
                break
            current_face = int(result["face_id"])
            current_barycentric = np.asarray(result["barycentric"])
            current_reference = np.asarray(result["direction"])
            points.append(np.asarray(result["point"]))
            faces.append(current_face)
            weights.append(current_barycentric.copy())
            confidence.append(float(result["confidence"]))
            crossings += int(result["crossing_count"])
        return {
            "surface_points": np.asarray(points),
            "face_ids": np.asarray(faces, dtype=np.int64),
            "barycentric": np.asarray(weights),
            "direction_confidence": np.asarray(confidence),
            "crossing_count": crossings,
            "status": status,
        }

    forward = attempt(reference)
    expected = int(round(total_length_m / step_length_m)) + 1
    if len(forward["surface_points"]) == expected:
        return forward
    reverse = attempt(-reference)
    return reverse if len(reverse["surface_points"]) > len(forward["surface_points"]) else forward


def couple_occluder_adaptively(
    root_trace: np.ndarray,
    guide: np.ndarray,
    requested_handoff_m: float,
    bridge_spacing_m: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """在首次遮挡之后自适应选交接点，并以五次 Hermite C² 接回原尾段。"""

    clean = effective_polyline(guide)
    length = float(arc_lengths(clean)[-1])
    radius = 0.005
    maximum = max(0.036, length - radius - 0.001)
    handoff = min(max(float(requested_handoff_m), 0.060), maximum)
    config = CouplingConfig(
        handoff_m=handoff,
        bridge_spacing_m=bridge_spacing_m,
        kinematics_radius_m=radius,
    )
    kinematics = sample_kinematics(clean, handoff, radius)
    coupled, diagnostics = couple_strand(root_trace, clean, kinematics, config)
    diagnostics["requested_handoff_m"] = float(requested_handoff_m)
    diagnostics["actual_handoff_m"] = float(handoff)
    diagnostics["handoff_clamped"] = int(handoff + 1e-9 < requested_handoff_m)
    return coupled, diagnostics


def build_occlusion_first_candidate(inputs: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """逐根投影到最近 atlas、积分 30 mm 根段并自适应耦合原尾段。"""

    strands = inputs["strands"]
    occluders = inputs["occluders"]
    selected_ids = np.asarray(occluders["strand_ids"], dtype=np.int64)
    roots = np.stack([np.asarray(strands[int(index)])[0] for index in selected_ids])
    tangents = np.stack([
        effective_polyline(strands[int(index)])[2] - effective_polyline(strands[int(index)])[0]
        for index in selected_ids
    ])
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-15)
    chart_names = ("left", "right")
    projections = {}
    for chart_name in chart_names:
        closest, distance, face_id = trimesh.proximity.closest_point(inputs["meshes"][chart_name], roots)
        projections[chart_name] = {"point": closest, "distance": distance, "face_id": face_id}
    distance_matrix = np.stack([projections[name]["distance"] for name in chart_names])
    chart_index = np.argmin(distance_matrix, axis=0)
    chart_id = np.where(chart_index == 0, -1, 1).astype(np.int8)
    projection_distance = np.min(distance_matrix, axis=0)
    caches = {
        name: prepare_chart_direction_field(inputs["meshes"][name], inputs["fields"][name])
        for name in chart_names
    }
    traces: list[np.ndarray | None] = [None] * len(selected_ids)
    trace_status: list[str] = [""] * len(selected_ids)
    for local_id, name_id in enumerate(chart_index):
        name = chart_names[int(name_id)]
        trace = integrate_projected_root_trace(
            caches[name], projections[name]["point"][local_id],
            int(projections[name]["face_id"][local_id]), tangents[local_id],
            args.trace_step_mm / 1000.0, args.trace_length_mm / 1000.0,
        )
        trace_status[local_id] = str(trace["status"])
        expected = int(round(args.trace_length_mm / args.trace_step_mm)) + 1
        if len(trace["surface_points"]) != expected:
            continue
        lifted, _ = lift_root_traces(
            [trace], caches[name], args.trace_length_mm / 1000.0,
            args.root_height_mm / 1000.0, args.peak_height_mm / 1000.0,
        )
        displacement = lifted[0] - lifted[0, 0]
        traces[local_id] = roots[local_id] + displacement
    merged = list(strands)
    local_strands: list[np.ndarray] = []
    processed_ids: list[int] = []
    diagnostics: list[dict[str, float | int]] = []
    failed_coupling: list[int] = []
    earliest = np.asarray(occluders["earliest_arclength_m"], dtype=np.float64)
    for local_id, strand_id in enumerate(selected_ids):
        if traces[local_id] is None:
            continue
        requested = max(args.minimum_handoff_mm / 1000.0, earliest[local_id] + args.post_occlusion_margin_mm / 1000.0)
        try:
            coupled, item = couple_occluder_adaptively(
                traces[local_id], strands[int(strand_id)], requested,
                args.bridge_spacing_mm / 1000.0,
            )
        except (ValueError, np.linalg.LinAlgError):
            failed_coupling.append(int(strand_id))
            continue
        merged[int(strand_id)] = coupled
        local_strands.append(coupled)
        processed_ids.append(int(strand_id))
        diagnostics.append(item)
    return {
        "merged": merged,
        "local": local_strands,
        "selected_ids": selected_ids,
        "processed_ids": np.asarray(processed_ids, dtype=np.int64),
        "chart_id": chart_id,
        "projection_distance_m": projection_distance,
        "trace_status": np.asarray(trace_status),
        "diagnostics": diagnostics,
        "failed_coupling_ids": np.asarray(failed_coupling, dtype=np.int64),
    }


def evaluate_step10b_candidate(
    inputs: dict[str, object], candidate: dict[str, object], args: argparse.Namespace
) -> dict[str, object]:
    """检查精确 10k、未选区不变、连续性、穿模和正面绝对发缝可见率。"""

    original = inputs["strands"]
    merged = candidate["merged"]
    processed = set(candidate["processed_ids"].tolist())
    unselected_max = 0.0
    for strand_id, (before, after) in enumerate(zip(original, merged)):
        if strand_id in processed:
            continue
        if before.shape != after.shape:
            unselected_max = float("inf")
            break
        unselected_max = max(unselected_max, float(np.max(np.abs(before - after))))
    before_points = np.concatenate([effective_polyline(original[index]) for index in processed])
    after_points = np.concatenate(candidate["local"])
    baseline_penetration = int(np.sum(surface_signed_heights(inputs["head_tm"], before_points) < -1e-5))
    candidate_penetration = int(np.sum(surface_signed_heights(inputs["head_tm"], after_points) < -1e-5))
    config = VisibilityConfig()
    camera = load_view_camera(args, "front")
    head_depth, origins, direction = build_head_depth_map(
        inputs["head_o3d"], camera["calib"], camera["toward_camera"], config.image_size
    )
    visibility = {}
    rasters = {}
    region = inputs["seg"] & inputs["parting"]
    for name, strands in (("baseline", original), ("candidate", merged)):
        segments = collect_projected_segments(strands, camera["calib"], config.image_size, config.physical_sampling_m)
        raster = rasterize_hair_view(segments, head_depth, origins, direction, config)
        occupied = int(np.sum(np.asarray(raster["occupancy"]) & region))
        visibility[name] = {
            "occupied_parting_pixels": occupied,
            "parting_region_pixels": int(np.sum(region)),
            "scalp_visibility_ratio": float(1.0 - occupied / max(np.sum(region), 1)),
        }
        rasters[name] = raster
    angles = np.asarray([item["handoff_angle_deg"] for item in candidate["diagnostics"]])
    jumps = np.asarray([item["handoff_curvature_jump_inv_m"] for item in candidate["diagnostics"]])
    requested = np.asarray([item["requested_handoff_m"] for item in candidate["diagnostics"]])
    actual = np.asarray([item["actual_handoff_m"] for item in candidate["diagnostics"]])
    numeric_gates = {
        "exactly_10000_strands": len(merged) == 10000,
        "processed_occluder_recall_ge_95pct": len(processed) / max(len(candidate["selected_ids"]), 1) >= 0.95,
        "unprocessed_strands_unchanged": unselected_max == 0.0,
        "handoff_angle_max_le_10deg": bool(len(angles) and np.max(angles) <= 10.0),
        "no_new_surface_penetration": candidate_penetration <= baseline_penetration,
        "absolute_front_scalp_visibility_ge_68pct": visibility["candidate"]["scalp_visibility_ratio"] >= 0.68,
    }
    return {
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "processed_occluder_recall": float(len(processed) / max(len(candidate["selected_ids"]), 1)),
        "unprocessed_point_max_abs_difference_m": unselected_max,
        "handoff_angle_deg": {
            "q50": float(np.quantile(angles, 0.50)), "q95": float(np.quantile(angles, 0.95)), "max": float(np.max(angles)),
        },
        "handoff_curvature_jump_inv_m": {
            "q50": float(np.quantile(jumps, 0.50)), "q95": float(np.quantile(jumps, 0.95)), "max": float(np.max(jumps)),
        },
        "handoff_m": {
            "requested_q95": float(np.quantile(requested, 0.95)),
            "actual_q95": float(np.quantile(actual, 0.95)),
            "clamped_count": int(np.sum(actual + 1e-9 < requested)),
        },
        "penetration_points": {
            "baseline_selected": baseline_penetration,
            "candidate_selected": candidate_penetration,
            "added": max(candidate_penetration - baseline_penetration, 0),
        },
        "front_visibility": visibility,
        "rasters": rasters,
    }


def render_step10b_preview(
    output_dir: Path,
    inputs: dict[str, object],
    candidate: dict[str, object],
    evaluation: dict[str, object],
    calib: np.ndarray,
) -> dict[str, str]:
    """用 raw_img.png 输出被处理发丝和绝对发缝占据的前后对比。"""

    processed = candidate["processed_ids"]
    side_by_id = dict(zip(candidate["selected_ids"].tolist(), candidate["chart_id"].tolist()))
    sides = np.asarray([side_by_id[int(index)] for index in processed], dtype=np.int8)
    strand_path = output_dir / "processed_before_after_on_raw_img.png"
    render_front_preview(
        strand_path, inputs["image"], calib,
        [inputs["strands"][int(index)] for index in processed], candidate["local"], sides,
        titles=("baseline true occluders", "Step 10B adaptive Connection-LB"),
    )
    panels = []
    for name, color in (("baseline", (255, 170, 30)), ("candidate", (40, 230, 80))):
        base = inputs["image"].copy()
        occupancy = np.asarray(evaluation["rasters"][name]["occupancy"], dtype=bool)
        overlay = base.copy()
        overlay[occupancy & inputs["parting"] & inputs["seg"]] = color
        panel = cv2.addWeighted(base, 0.60, overlay, 0.40, 0.0)
        cv2.putText(panel, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    visibility_path = output_dir / "absolute_parting_visibility_on_raw_img.png"
    cv2.imwrite(str(visibility_path), np.concatenate(panels, axis=1))
    return {"processed_strands_preview": str(strand_path), "absolute_visibility_preview": str(visibility_path)}


def run_step10b_density_rebuild(args: argparse.Namespace) -> dict[str, object]:
    """构建并评估遮挡优先的完整 10k 隔离候选。"""

    inputs = load_step10b_inputs(args)
    candidate = build_occlusion_first_candidate(inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "full_10k_occlusion_first_connection_lb.ply"
    local_path = args.output_dir / "local_occlusion_first_layer.ply"
    write_ordered_strands(candidate["merged"], candidate_path)
    write_ordered_strands(candidate["local"], local_path)
    evaluation = evaluate_step10b_candidate(inputs, candidate, args)
    camera = load_view_camera(args, "front")
    outputs = render_step10b_preview(args.output_dir, inputs, candidate, evaluation, camera["calib"])
    outputs.update({"candidate_ply": str(candidate_path), "local_ply": str(local_path)})
    diagnostics = candidate["diagnostics"]
    np.savez_compressed(
        args.output_dir / "step10b_selection.npz",
        selected_occluder_ids=candidate["selected_ids"],
        processed_ids=candidate["processed_ids"],
        chart_id=candidate["chart_id"],
        atlas_projection_distance_m=candidate["projection_distance_m"],
        trace_status=candidate["trace_status"],
        handoff_angle_deg=np.asarray([item["handoff_angle_deg"] for item in diagnostics]),
        requested_handoff_m=np.asarray([item["requested_handoff_m"] for item in diagnostics]),
        actual_handoff_m=np.asarray([item["actual_handoff_m"] for item in diagnostics]),
    )
    outputs["selection_data"] = str(args.output_dir / "step10b_selection.npz")
    metrics = {key: value for key, value in evaluation.items() if key != "rasters"}
    report = {
        "image_id": args.image_id,
        "step": "step_10b_occlusion_first_density_rebuild",
        "purpose": "用真实深度可见遮挡集合替代固定 256 根选择，并以首次遮挡位置控制自适应交接",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "visible_occluders": str(args.visible_occluders.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
        },
        "config": {
            "trace_step_mm": args.trace_step_mm,
            "trace_length_mm": args.trace_length_mm,
            "minimum_handoff_mm": args.minimum_handoff_mm,
            "post_occlusion_margin_mm": args.post_occlusion_margin_mm,
            "bridge_spacing_mm": args.bridge_spacing_mm,
            "root_policy": "preserve original 3D root; atlas projection supplies Connection-LB displacement field",
            "selection_policy": "all Step 10A baseline visible occluders",
        },
        "selection": {
            "true_occluder_count": int(len(candidate["selected_ids"])),
            "processed_count": int(len(candidate["processed_ids"])),
            "trace_incomplete_count": int(np.sum(candidate["trace_status"] != "complete")),
            "failed_coupling_count": int(len(candidate["failed_coupling_ids"])),
            "atlas_projection_distance_m": {
                "q50": float(np.quantile(candidate["projection_distance_m"], 0.50)),
                "q95": float(np.quantile(candidate["projection_distance_m"], 0.95)),
                "max": float(np.max(candidate["projection_distance_m"])),
            },
        },
        "metrics": metrics,
        "numeric_gates": evaluation["numeric_gates"],
        "passed_numeric_gates": evaluation["passed_numeric_gates"],
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": outputs,
        "parent_step_report": str(args.step10a_report.resolve()),
    }
    report_path = args.output_dir / "step_10b_occlusion_first_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step10b_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 10B 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    connection = base / "connection_lb_parting"
    step2 = connection / "step_02_cut_atlas"
    step5 = connection / "step_05_observed_field"
    step10a = connection / "step_10_occlusion_density/step_10a_visible_occluders"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step10a-report", type=Path, default=step10a / "step_10a_visible_occluder_report.json")
    parser.add_argument("--visible-occluders", type=Path, default=step10a / "10k_baseline_visible_occluders.npz")
    parser.add_argument("--full-10k-baseline", type=Path, default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply")
    parser.add_argument("--left-atlas", type=Path, default=step2 / "left_atlas.obj")
    parser.add_argument("--right-atlas", type=Path, default=step2 / "right_atlas.obj")
    parser.add_argument("--observed-fields", type=Path, default=step5 / "observed_fields.npz")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--trace-step-mm", type=float, default=1.0)
    parser.add_argument("--trace-length-mm", type=float, default=30.0)
    parser.add_argument("--root-height-mm", type=float, default=0.5)
    parser.add_argument("--peak-height-mm", type=float, default=4.0)
    parser.add_argument("--minimum-handoff-mm", type=float, default=60.0)
    parser.add_argument("--post-occlusion-margin-mm", type=float, default=20.0)
    parser.add_argument("--bridge-spacing-mm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=connection / "step_10_occlusion_density/step_10b_occlusion_first_candidate")
    return parser


def step10b_main() -> None:
    """运行 Step 10B 并打印绝对发缝门禁摘要。"""

    report = run_step10b_density_rebuild(build_step10b_arg_parser().parse_args())
    print(json.dumps({
        "passed_numeric_gates": report["passed_numeric_gates"],
        "numeric_gates": report["numeric_gates"],
        "selection": report["selection"],
        "front_visibility": report["metrics"]["front_visibility"],
        "outputs": report["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step10b_main()
