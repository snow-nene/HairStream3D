#!/usr/bin/env python3
"""Step 9A：在既有完整 10k 基线上构造 Connection-LB 局部耦合候选。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import (
    build_curve_lateral_frame,
    curve_coordinates,
    read_ordered_strands,
    write_ordered_strands,
)
from scripts.recon_3d.couple_parting_outer_strands import (
    CouplingConfig,
    build_match_cost,
    couple_strand,
    evaluate_coupling,
    identify_affected_guides,
    match_guides_global,
    render_front_preview,
    surface_signed_heights,
)
from scripts.recon_3d.solve_parting_bank_only_field import calibration_from_param


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step9_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 6/8 已通过并读取完整 10k 基线及耦合输入。"""

    step6_report = json.loads(args.step6_report.read_text(encoding="utf-8"))
    step8_report = json.loads(args.step8_report.read_text(encoding="utf-8"))
    if not step6_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 6 数值门禁未通过")
    if not step8_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 8 数值门禁未通过")
    if step8_report.get("human_review_status") != "accepted":
        raise RuntimeError("Step 8 尚未被人工标记为 accepted")
    with np.load(args.root_traces, allow_pickle=False) as archive:
        root_points = np.asarray(archive["points"], dtype=np.float64)
        chart_id = np.asarray(archive["chart_id"], dtype=np.int8)
    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
    guides = read_ordered_strands(args.full_10k_baseline)
    if len(guides) != 10000:
        raise ValueError(f"完整基线应为 10000 根，实际为 {len(guides)}")
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    parting_mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    if image is None or parting_mask is None or seg is None:
        raise FileNotFoundError("无法读取 raw_img、parting mask 或 front seg")
    calib = calibration_from_param(args.front_calib)
    calib[2] = calibration_from_param(args.front_depth_calib)[2]
    head = trimesh.load(str(args.head_mesh), process=True)
    if isinstance(head, trimesh.Scene):
        head = trimesh.util.concatenate(tuple(head.geometry.values()))
    if not isinstance(head, trimesh.Trimesh):
        raise ValueError("头模不是三角网格")
    return {
        "step6_report": step6_report,
        "step8_report": step8_report,
        "root_points": root_points,
        "chart_id": chart_id,
        "curve": curve,
        "guides": guides,
        "image": image,
        "parting_mask": parting_mask,
        "seg": seg,
        "calib": calib,
        "head": head,
    }


def repair_invalid_matches(
    root_points: np.ndarray,
    root_side: np.ndarray,
    curve: np.ndarray,
    lateral: np.ndarray,
    guide_data: dict[str, object],
    matched_ids: np.ndarray,
    match_cost: np.ndarray,
    local: list[np.ndarray],
    diagnostics: list[dict[str, float | int]],
    head: trimesh.Trimesh,
    config: CouplingConfig,
    maximum_trials: int = 256,
) -> list[dict[str, float | int]]:
    """仅为交接角或穿模越界发束重新选择同岸未占用尾段。"""

    _, _, root_curve_index = curve_coordinates(root_points[:, 0], curve, lateral)
    valid = np.asarray(guide_data["valid"], dtype=bool)
    guide_side = np.asarray(guide_data["side"], dtype=np.int8)
    clean_guides = guide_data["clean_guides"]
    kinematics = guide_data["kinematics"]
    used = set(int(index) for index in matched_ids)
    repairs: list[dict[str, float | int]] = []
    for trace_id in range(len(root_points)):
        strand = local[trace_id]
        item = diagnostics[trace_id]
        penetration = int(
            np.sum(
                surface_signed_heights(head, strand)
                < config.penetration_tolerance_m
            )
        )
        if (
            float(item["handoff_angle_deg"])
            <= config.maximum_handoff_angle_deg
            and penetration == 0
        ):
            continue
        previous_id = int(matched_ids[trace_id])
        candidates = np.flatnonzero(
            valid & (guide_side == int(root_side[trace_id]))
        )
        candidates = np.asarray(
            [index for index in candidates if int(index) not in used],
            dtype=np.int64,
        )
        matrix = build_match_cost(
            root_points[trace_id : trace_id + 1],
            root_curve_index[trace_id : trace_id + 1],
            candidates,
            guide_data,
            len(curve),
        )[0]
        replacement = None
        for candidate_pos in np.argsort(matrix)[: int(maximum_trials)]:
            guide_id = int(candidates[int(candidate_pos)])
            candidate_strand, candidate_item = couple_strand(
                root_points[trace_id],
                clean_guides[guide_id],
                kinematics[guide_id],
                config,
            )
            candidate_penetration = int(
                np.sum(
                    surface_signed_heights(head, candidate_strand)
                    < config.penetration_tolerance_m
                )
            )
            if (
                float(candidate_item["handoff_angle_deg"])
                <= config.maximum_handoff_angle_deg
                and candidate_penetration == 0
            ):
                replacement = (
                    guide_id,
                    float(matrix[int(candidate_pos)]),
                    candidate_strand,
                    candidate_item,
                )
                break
        if replacement is None:
            raise RuntimeError(
                f"root trace {trace_id} 在 {maximum_trials} 个同岸候选中无有效替换"
            )
        guide_id, cost, candidate_strand, candidate_item = replacement
        used.remove(previous_id)
        used.add(guide_id)
        matched_ids[trace_id] = guide_id
        match_cost[trace_id] = cost
        local[trace_id] = candidate_strand
        diagnostics[trace_id] = candidate_item
        repairs.append(
            {
                "root_trace_id": trace_id,
                "previous_guide_id": previous_id,
                "replacement_guide_id": guide_id,
                "previous_handoff_angle_deg": float(item["handoff_angle_deg"]),
                "previous_penetration_points": penetration,
                "replacement_handoff_angle_deg": float(
                    candidate_item["handoff_angle_deg"]
                ),
                "replacement_match_cost": cost,
            }
        )
    return repairs


def run_step9_10k_coupling(args: argparse.Namespace) -> dict[str, object]:
    """对 10k 基线做 256 根局部替换，并执行 Step 7 同级耦合门禁。"""

    inputs = load_step9_inputs(args)
    config = CouplingConfig(
        guide_root_influence_m=args.guide_root_influence_mm / 1000.0,
        handoff_m=args.handoff_mm / 1000.0,
        bridge_spacing_m=args.bridge_spacing_mm / 1000.0,
        influence_dilation_px=args.influence_dilation_px,
    )
    root_points = np.asarray(inputs["root_points"])
    root_side = np.asarray(inputs["chart_id"], dtype=np.int8)
    curve = np.asarray(inputs["curve"])
    lateral = build_curve_lateral_frame(curve, root_points[:, 0], root_side)
    guide_data = identify_affected_guides(
        inputs["guides"],
        curve,
        lateral,
        inputs["parting_mask"],
        inputs["seg"],
        inputs["calib"],
        inputs["image"].shape,
        config,
    )
    matched_ids, match_cost = match_guides_global(
        root_points, root_side, curve, lateral, guide_data
    )
    clean_guides = guide_data["clean_guides"]
    kinematics = guide_data["kinematics"]
    local: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int]] = []
    for trace, guide_id in zip(root_points, matched_ids):
        strand, item = couple_strand(
            trace,
            clean_guides[int(guide_id)],
            kinematics[int(guide_id)],
            config,
        )
        local.append(strand)
        diagnostics.append(item)
    repairs = repair_invalid_matches(
        root_points,
        root_side,
        curve,
        lateral,
        guide_data,
        matched_ids,
        match_cost,
        local,
        diagnostics,
        inputs["head"],
        config,
    )
    original = inputs["guides"]
    merged = list(original)
    for guide_id, strand in zip(matched_ids, local):
        merged[int(guide_id)] = strand
    metrics = evaluate_coupling(
        original,
        merged,
        matched_ids,
        local,
        diagnostics,
        inputs["head"],
        config,
    )
    exact_count = len(merged) == 10000
    unselected_count = len(merged) - len(matched_ids)
    metrics["numeric_gates"]["exactly_10000_strands"] = exact_count
    metrics["numeric_gates"]["exactly_9744_unselected_strands"] = unselected_count == 9744
    metrics["passed_numeric_gates"] = bool(all(metrics["numeric_gates"].values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.output_dir / "full_10k_before.ply"
    local_path = args.output_dir / "local_connection_lb_256.ply"
    merged_path = args.output_dir / "full_10k_connection_lb.ply"
    shutil.copy2(args.full_10k_baseline, before_path)
    write_ordered_strands(local, local_path)
    write_ordered_strands(merged, merged_path)
    preview_path = args.output_dir / "full_10k_selected_before_after_on_raw_img.png"
    render_front_preview(
        preview_path,
        inputs["image"],
        inputs["calib"],
        [original[int(index)] for index in matched_ids],
        local,
        root_side,
        titles=("10k baseline selected", "10k Connection-LB"),
    )
    valid = np.asarray(guide_data["valid"], dtype=bool)
    guide_side = np.asarray(guide_data["side"], dtype=np.int8)
    bridge_chord = np.asarray([item["bridge_chord_m"] for item in diagnostics])
    report = {
        "image_id": args.image_id,
        "step": "step_09a_10k_outer_coupling",
        "purpose": "在完整 10000 根基线上仅替换发缝影响区 256 根，验证规模放大前的局部耦合",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "root_traces": str(args.root_traces.resolve()),
            "front_image": str(args.front_image.resolve()),
        },
        "config": {
            "total_strands": 10000,
            "replacement_strands": 256,
            "guide_root_influence_mm": args.guide_root_influence_mm,
            "handoff_mm": args.handoff_mm,
            "bridge_spacing_mm": args.bridge_spacing_mm,
            "matching": "same-bank Hungarian global unique assignment",
            "coupling": "quintic Hermite position+tangent+curvature",
        },
        "selection": {
            "selected": len(matched_ids),
            "unselected": unselected_count,
            "selected_side_counts": {
                str(bank): int(np.sum(root_side == bank)) for bank in (-1, 1)
            },
            "valid_candidate_side_counts": {
                str(bank): int(np.sum(valid & (guide_side == bank)))
                for bank in (-1, 1)
            },
            "selected_guide_ids": matched_ids.tolist(),
        },
        "matching": {
            "cost": {
                "q50": float(np.quantile(match_cost, 0.50)),
                "q95": float(np.quantile(match_cost, 0.95)),
                "max": float(np.max(match_cost)),
            },
            "bridge_chord_m": {
                "q50": float(np.quantile(bridge_chord, 0.50)),
                "q95": float(np.quantile(bridge_chord, 0.95)),
                "max": float(np.max(bridge_chord)),
            },
            "invalid_match_repairs": repairs,
            "invalid_match_repair_count": len(repairs),
        },
        "metrics": metrics,
        "numeric_gates": metrics["numeric_gates"],
        "passed_numeric_gates": metrics["passed_numeric_gates"],
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "before_ply": str(before_path),
            "local_ply": str(local_path),
            "merged_ply": str(merged_path),
            "front_raw_preview": str(preview_path),
        },
        "parent_step_report": str(args.step8_report.resolve()),
    }
    np.savez_compressed(
        args.output_dir / "coupling_selection_10k.npz",
        selected_guide_ids=matched_ids,
        root_trace_ids=np.arange(len(root_points), dtype=np.int32),
        side=root_side,
        match_cost=match_cost,
        handoff_angle_deg=np.asarray(
            [item["handoff_angle_deg"] for item in diagnostics]
        ),
        handoff_curvature_jump_inv_m=np.asarray(
            [item["handoff_curvature_jump_inv_m"] for item in diagnostics]
        ),
    )
    report_path = args.output_dir / "step_09a_10k_outer_coupling_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_step9_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 9A 命令行参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument(
        "--step6-report",
        type=Path,
        default=base / "connection_lb_parting/step_06_root_traces/step_06_root_traces_report.json",
    )
    parser.add_argument(
        "--step8-report",
        type=Path,
        default=base / "connection_lb_parting/step_08_render_gate/step_08_render_gate_report.json",
    )
    parser.add_argument(
        "--root-traces",
        type=Path,
        default=base / "connection_lb_parting/step_06_root_traces/root_traces.npz",
    )
    parser.add_argument(
        "--atlas-data",
        type=Path,
        default=base / "connection_lb_parting/step_02_cut_atlas/cut_atlas_data.npz",
    )
    parser.add_argument(
        "--full-10k-baseline",
        type=Path,
        default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply",
    )
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument(
        "--parting-mask",
        type=Path,
        default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png",
    )
    parser.add_argument(
        "--front-calib",
        type=Path,
        default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy",
    )
    parser.add_argument(
        "--front-depth-calib",
        type=Path,
        default=DEFAULT_ROOT / "maps/param/front.npy",
    )
    parser.add_argument("--guide-root-influence-mm", type=float, default=35.0)
    parser.add_argument("--handoff-mm", type=float, default=60.0)
    parser.add_argument("--bridge-spacing-mm", type=float, default=1.0)
    parser.add_argument("--influence-dilation-px", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "connection_lb_parting/step_09_10k_scale/step_09a_outer_coupling",
    )
    return parser


def step9_main() -> None:
    """运行 Step 9A 并打印门禁摘要。"""

    report = run_step9_10k_coupling(build_step9_arg_parser().parse_args())
    print(
        json.dumps(
            {
                "passed_numeric_gates": report["passed_numeric_gates"],
                "numeric_gates": report["numeric_gates"],
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    step9_main()
