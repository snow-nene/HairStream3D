#!/usr/bin/env python3
"""Step 10A：提取完整 10k 中真正深度可见且遮挡发缝的发丝。"""

from __future__ import annotations

import argparse
import hashlib
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
MODEL_NAMES = ("10k_baseline", "10k_connection_lb")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_step10a_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 9B 已被拒绝，并读取两组 10k、原图和正面几何输入。"""

    parent = json.loads(args.step9b_report.read_text(encoding="utf-8"))
    if parent.get("human_review_status") != "rejected":
        raise RuntimeError("Step 10A 只用于排查已被人工拒绝的 Step 9B")
    models = {
        "10k_baseline": read_ordered_strands(args.full_10k_baseline),
        "10k_connection_lb": read_ordered_strands(args.full_10k_candidate),
    }
    if any(len(strands) != 10000 for strands in models.values()):
        raise ValueError("Step 10A 的两组输入都必须严格包含 10000 根发丝")
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    parting = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or seg is None or parting is None:
        raise FileNotFoundError("缺少 raw_img.png、front seg 或发缝 mask")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    seg = cv2.resize(seg, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
    parting = cv2.resize(parting, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    head = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not head.has_vertices() or not head.has_triangles():
        raise ValueError("头模为空")
    selection = np.load(args.step9a_selection, allow_pickle=False)
    selected = np.asarray(selection["selected_guide_ids"], dtype=np.int64)
    if len(selected) != 256 or len(np.unique(selected)) != 256:
        raise ValueError("Step 9A 选择集合必须是 256 个唯一 guide ID")
    return {
        "parent": parent,
        "models": models,
        "image": image,
        "seg": seg,
        "parting": parting,
        "region": seg & parting,
        "head": head,
        "selected": selected,
    }


def extract_visible_occluders(
    strands: list[np.ndarray],
    calib: np.ndarray,
    head_depth: np.ndarray,
    ray_origins: np.ndarray,
    ray_direction: np.ndarray,
    region: np.ndarray,
    config: VisibilityConfig,
    prefix_length_m: float,
) -> dict[str, object]:
    """逐根做首交深度测试，并把 Step 9B 的 1 px 足迹归因到发缝像素。"""

    image_size = config.image_size
    offsets = np.array(
        [
            (dx, dy)
            for dy in range(-config.occupancy_dilation_px, config.occupancy_dilation_px + 1)
            for dx in range(-config.occupancy_dilation_px, config.occupancy_dilation_px + 1)
        ],
        dtype=np.int32,
    )
    strand_ids: list[int] = []
    pixel_sets: list[np.ndarray] = []
    prefix_pixel_sets: list[np.ndarray] = []
    earliest_arclength: list[float] = []
    root_pixels: list[np.ndarray] = []
    visible_sample_count: list[int] = []
    for strand_id, strand in enumerate(strands):
        try:
            sampled, arclength = resample_polyline(strand, config.physical_sampling_m)
        except ValueError:
            continue
        midpoint = 0.5 * (sampled[:-1] + sampled[1:])
        midpoint_s = 0.5 * (arclength[:-1] + arclength[1:])
        midpoint_px, _ = project_view(midpoint, calib, image_size)
        px = np.rint(midpoint_px[:, 0]).astype(np.int32)
        py = np.rint(midpoint_px[:, 1]).astype(np.int32)
        inside = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        hair_distance = np.sum(
            (midpoint[ids] - ray_origins[py[ids], px[ids]]) * ray_direction[None, :],
            axis=1,
        )
        visible_ids = ids[
            hair_distance
            <= head_depth[py[ids], px[ids]] + config.head_visibility_tolerance_m
        ]
        if not len(visible_ids):
            continue
        expanded_x = px[visible_ids, None] + offsets[None, :, 0]
        expanded_y = py[visible_ids, None] + offsets[None, :, 1]
        expanded_inside = (
            (expanded_x >= 0)
            & (expanded_x < image_size)
            & (expanded_y >= 0)
            & (expanded_y < image_size)
        )
        linear = expanded_y * image_size + expanded_x
        valid_linear = linear[expanded_inside]
        in_region = region.reshape(-1)[valid_linear]
        pixels = np.unique(valid_linear[in_region]).astype(np.int32)
        if not len(pixels):
            continue
        prefix_visible = visible_ids[midpoint_s[visible_ids] <= prefix_length_m]
        prefix_x = px[prefix_visible, None] + offsets[None, :, 0]
        prefix_y = py[prefix_visible, None] + offsets[None, :, 1]
        prefix_inside = (
            (prefix_x >= 0)
            & (prefix_x < image_size)
            & (prefix_y >= 0)
            & (prefix_y < image_size)
        )
        prefix_linear = prefix_y * image_size + prefix_x
        prefix_valid = prefix_linear[prefix_inside]
        prefix_pixels = np.unique(prefix_valid[region.reshape(-1)[prefix_valid]]).astype(np.int32)
        contributing_samples = []
        pixel_lookup = set(pixels.tolist())
        for sample_id in visible_ids:
            footprint = (
                (py[sample_id] + offsets[:, 1]) * image_size
                + px[sample_id]
                + offsets[:, 0]
            )
            if any(int(value) in pixel_lookup for value in footprint):
                contributing_samples.append(sample_id)
        root_px, _ = project_view(sampled[:1], calib, image_size)
        strand_ids.append(strand_id)
        pixel_sets.append(pixels)
        prefix_pixel_sets.append(prefix_pixels)
        earliest_arclength.append(float(midpoint_s[min(contributing_samples)]))
        root_pixels.append(root_px[0])
        visible_sample_count.append(len(visible_ids))
    return {
        "strand_ids": np.asarray(strand_ids, dtype=np.int32),
        "pixel_sets": pixel_sets,
        "prefix_pixel_sets": prefix_pixel_sets,
        "earliest_arclength_m": np.asarray(earliest_arclength, dtype=np.float32),
        "root_pixels": np.asarray(root_pixels, dtype=np.float32),
        "visible_sample_count": np.asarray(visible_sample_count, dtype=np.int32),
    }


def rank_occluder_contributions(extraction: dict[str, object], image_size: int) -> dict[str, np.ndarray]:
    """按共享像素的分数份额排序，确保总贡献等于被遮挡像素数。"""

    pixel_sets = extraction["pixel_sets"]
    frequency = np.zeros(image_size * image_size, dtype=np.int32)
    for pixels in pixel_sets:
        frequency[pixels] += 1
    contribution = np.asarray(
        [np.sum(1.0 / frequency[pixels]) for pixels in pixel_sets], dtype=np.float64
    )
    pixel_count = np.asarray([len(pixels) for pixels in pixel_sets], dtype=np.int32)
    prefix_pixel_count = np.asarray(
        [len(pixels) for pixels in extraction["prefix_pixel_sets"]], dtype=np.int32
    )
    strand_ids = np.asarray(extraction["strand_ids"])
    order = np.lexsort((strand_ids, -pixel_count, -contribution))
    return {
        "frequency": frequency.reshape(image_size, image_size),
        "contribution": contribution,
        "pixel_count": pixel_count,
        "prefix_pixel_count": prefix_pixel_count,
        "order": order.astype(np.int32),
        "rank": np.argsort(order).astype(np.int32) + 1,
    }


def _pack_pixel_sets(pixel_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(pixel_sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(pixels) for pixels in pixel_sets])
    values = np.concatenate(pixel_sets).astype(np.int32) if pixel_sets else np.empty(0, np.int32)
    return offsets, values


def render_occluder_diagnostics(
    output_dir: Path,
    image: np.ndarray,
    region: np.ndarray,
    extractions: dict[str, dict[str, object]],
    rankings: dict[str, dict[str, np.ndarray]],
    selected: np.ndarray,
) -> dict[str, str]:
    """在 raw_img.png 上输出真实遮挡归因、旧 256 召回和频次热图。"""

    outputs: dict[str, str] = {}
    selected_set = set(selected.tolist())
    panels = []
    for model in MODEL_NAMES:
        extraction = extractions[model]
        ranking = rankings[model]
        ids = np.asarray(extraction["strand_ids"])
        panel = image.copy()
        overlay = panel.copy()
        frequency = ranking["frequency"]
        overlay[(frequency > 0) & region] = (40, 80, 255)
        panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0.0)
        for strand_id, root in zip(ids, extraction["root_pixels"]):
            x, y = np.rint(root).astype(int)
            if 0 <= x < 512 and 0 <= y < 512:
                color = (50, 230, 50) if int(strand_id) in selected_set else (255, 190, 30)
                cv2.circle(panel, (x, y), 1, color, -1, cv2.LINE_AA)
        cv2.putText(panel, model, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
        path = output_dir / f"{model}_visible_occluders_on_raw_img.png"
        cv2.imwrite(str(path), panel)
        outputs[f"{model}_raw_overlay"] = str(path)
    compare_path = output_dir / "baseline_vs_candidate_visible_occluders_on_raw_img.png"
    cv2.imwrite(str(compare_path), np.concatenate(panels, axis=1))
    outputs["raw_comparison"] = str(compare_path)

    candidate_frequency = rankings["10k_connection_lb"]["frequency"].astype(np.float32)
    normalized = np.zeros_like(candidate_frequency, dtype=np.uint8)
    if np.any(candidate_frequency > 0):
        scale = np.percentile(candidate_frequency[candidate_frequency > 0], 95)
        normalized = np.clip(candidate_frequency / max(scale, 1.0) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heat_overlay = cv2.addWeighted(image, 0.58, heat, 0.42, 0.0)
    heat_overlay[~region] = image[~region]
    heat_path = output_dir / "candidate_occluder_frequency_heatmap_on_raw_img.png"
    cv2.imwrite(str(heat_path), heat_overlay)
    outputs["candidate_frequency_heatmap"] = str(heat_path)

    canvas = np.full((480, 720, 3), 248, dtype=np.uint8)
    for model, color in zip(MODEL_NAMES, ((255, 130, 30), (40, 180, 50))):
        ranking = rankings[model]
        cumulative = np.cumsum(ranking["contribution"][ranking["order"]])
        cumulative /= max(float(cumulative[-1]) if len(cumulative) else 1.0, 1.0)
        if len(cumulative):
            xs = 55 + np.arange(len(cumulative)) / max(len(cumulative) - 1, 1) * 630
            ys = 425 - cumulative * 370
            points = np.rint(np.column_stack([xs, ys])).astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (55, 425), (685, 425), (30, 30, 30), 1)
    cv2.line(canvas, (55, 425), (55, 55), (30, 30, 30), 1)
    cv2.putText(canvas, "ranked strand count", (270, 463), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "pixel contribution coverage", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    curve_path = output_dir / "occluder_contribution_cumulative.png"
    cv2.imwrite(str(curve_path), canvas)
    outputs["contribution_curve"] = str(curve_path)
    return outputs


def run_step10a_occluder_audit(args: argparse.Namespace) -> dict[str, object]:
    """运行只读的 10k 发缝真实遮挡归因审计。"""

    config = VisibilityConfig(physical_sampling_m=args.sampling_mm / 1000.0)
    inputs = load_step10a_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = load_view_camera(args, "front")
    head_depth, ray_origins, ray_direction = build_head_depth_map(
        inputs["head"], camera["calib"], camera["toward_camera"], config.image_size
    )
    extractions = {
        model: extract_visible_occluders(
            inputs["models"][model], camera["calib"], head_depth, ray_origins,
            ray_direction, inputs["region"], config, args.prefix_mm / 1000.0,
        )
        for model in MODEL_NAMES
    }
    rankings = {
        model: rank_occluder_contributions(extractions[model], config.image_size)
        for model in MODEL_NAMES
    }
    selected_set = set(inputs["selected"].tolist())
    id_sets = {model: set(extractions[model]["strand_ids"].tolist()) for model in MODEL_NAMES}
    per_model: dict[str, object] = {}
    for model in MODEL_NAMES:
        extraction = extractions[model]
        ranking = rankings[model]
        ids = np.asarray(extraction["strand_ids"])
        top_count = min(256, len(ids))
        top_ids = set(ids[ranking["order"][:top_count]].tolist())
        occupied = int(np.sum(ranking["frequency"] > 0))
        top_pixels: set[int] = set()
        for index in ranking["order"][:top_count]:
            top_pixels.update(extraction["pixel_sets"][index].tolist())
        selected_pixels: set[int] = set()
        for index, strand_id in enumerate(ids):
            if int(strand_id) in selected_set:
                selected_pixels.update(extraction["pixel_sets"][index].tolist())
        per_model[model] = {
            "visible_occluder_count": int(len(ids)),
            "prefix_60mm_occluder_count": int(np.sum(ranking["prefix_pixel_count"] > 0)),
            "late_entry_occluder_count": int(np.sum(ranking["prefix_pixel_count"] == 0)),
            "occupied_parting_pixels": occupied,
            "step9a_selected_occluder_count": int(len(id_sets[model] & selected_set)),
            "step9a_selected_occluder_recall": float(len(id_sets[model] & selected_set) / max(len(ids), 1)),
            "step9a_selected_pixel_coverage": float(len(selected_pixels) / max(occupied, 1)),
            "top256_contribution_pixel_coverage": float(len(top_pixels) / max(occupied, 1)),
            "top256_overlap_with_step9a": int(len(top_ids & selected_set)),
            "contribution_sum": float(np.sum(ranking["contribution"])),
        }
        pixel_offsets, pixel_values = _pack_pixel_sets(extraction["pixel_sets"])
        prefix_offsets, prefix_values = _pack_pixel_sets(extraction["prefix_pixel_sets"])
        np.savez_compressed(
            args.output_dir / f"{model}_visible_occluders.npz",
            strand_ids=ids,
            root_pixels=np.asarray(extraction["root_pixels"]),
            earliest_arclength_m=np.asarray(extraction["earliest_arclength_m"]),
            visible_sample_count=np.asarray(extraction["visible_sample_count"]),
            visible_pixel_count=ranking["pixel_count"],
            prefix_visible_pixel_count=ranking["prefix_pixel_count"],
            fractional_pixel_contribution=ranking["contribution"],
            rank=ranking["rank"],
            pixel_offsets=pixel_offsets,
            pixel_values=pixel_values,
            prefix_pixel_offsets=prefix_offsets,
            prefix_pixel_values=prefix_values,
        )
    outputs = render_occluder_diagnostics(
        args.output_dir, inputs["image"], inputs["region"], extractions,
        rankings, inputs["selected"],
    )
    outputs.update({
        f"{model}_data": str(args.output_dir / f"{model}_visible_occluders.npz")
        for model in MODEL_NAMES
    })
    baseline_frequency = rankings["10k_baseline"]["frequency"]
    candidate_frequency = rankings["10k_connection_lb"]["frequency"]
    numeric_gates = {
        "both_inputs_exactly_10000_strands": all(len(inputs["models"][model]) == 10000 for model in MODEL_NAMES),
        "all_reported_occluders_have_pixels": all(np.all(rankings[model]["pixel_count"] > 0) for model in MODEL_NAMES),
        "occluder_ids_unique": all(len(id_sets[model]) == len(extractions[model]["strand_ids"]) for model in MODEL_NAMES),
        "contribution_conserves_occupied_pixels": all(
            abs(per_model[model]["contribution_sum"] - per_model[model]["occupied_parting_pixels"]) <= 1e-6
            for model in MODEL_NAMES
        ),
        "candidate_does_not_reduce_parting_occlusion": int(np.sum(candidate_frequency > 0)) >= int(np.sum(baseline_frequency > 0)),
    }
    report = {
        "image_id": args.image_id,
        "step": "step_10a_visible_parting_occluder_audit",
        "purpose": "解释 Step 9B 人工拒绝：识别真正通过首交深度测试并覆盖发缝像素的 10k 发丝；不修改几何",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "full_10k_candidate": str(args.full_10k_candidate.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
            "baseline_sha256_before": _sha256(args.full_10k_baseline),
            "candidate_sha256_before": _sha256(args.full_10k_candidate),
        },
        "config": {
            **config.__dict__,
            "prefix_length_m": args.prefix_mm / 1000.0,
            "attribution_region": "parting_mask AND front seg",
            "attribution_footprint": "same 1px dilation as Step 9B",
        },
        "metrics": {
            "parting_region_pixel_count": int(np.sum(inputs["region"])),
            "models": per_model,
            "set_transition": {
                "persistent_occluders": int(len(id_sets["10k_baseline"] & id_sets["10k_connection_lb"])),
                "removed_occluders": int(len(id_sets["10k_baseline"] - id_sets["10k_connection_lb"])),
                "introduced_occluders": int(len(id_sets["10k_connection_lb"] - id_sets["10k_baseline"])),
            },
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "diagnostic_interpretation": "candidate_does_not_reduce_parting_occlusion 为真代表复现失败原因，不代表方案质量通过",
        "requires_human_review": True,
        "human_review_status": "pending",
        "geometry_modified": False,
        "outputs": outputs,
        "parent_step_report": str(args.step9b_report.resolve()),
    }
    report_path = args.output_dir / "step_10a_visible_occluder_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_step10a_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 10A 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    step9 = base / "connection_lb_parting/step_09_10k_scale"
    step9a = step9 / "step_09a_outer_coupling_repaired"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--step9b-report", type=Path, default=step9 / "step_09b_render_gate/step_09b_10k_render_gate_report.json")
    parser.add_argument("--full-10k-baseline", type=Path, default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply")
    parser.add_argument("--full-10k-candidate", type=Path, default=step9a / "full_10k_connection_lb.ply")
    parser.add_argument("--step9a-selection", type=Path, default=step9a / "coupling_selection_10k.npz")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--parting-mask", type=Path, default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--sampling-mm", type=float, default=2.0)
    parser.add_argument("--prefix-mm", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=base / "connection_lb_parting/step_10_occlusion_density/step_10a_visible_occluders")
    return parser


def step10a_main() -> None:
    """运行 Step 10A 并打印真实遮挡集合摘要。"""

    report = run_step10a_occluder_audit(build_step10a_arg_parser().parse_args())
    print(json.dumps({
        "passed_numeric_gates": report["passed_numeric_gates"],
        "metrics": report["metrics"],
        "outputs": report["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    step10a_main()
