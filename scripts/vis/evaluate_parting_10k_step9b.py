#!/usr/bin/env python3
"""Step 9B：完整 10k 基线与 Connection-LB 10k 候选的四视图规模门禁。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import read_ordered_strands
from scripts.vis.evaluate_parting_step8 import (
    VIEWS,
    VisibilityConfig,
    build_head_depth_map,
    collect_projected_segments,
    evaluate_model_view,
    load_view_camera,
    rasterize_hair_view,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
MODELS = ("10k_baseline", "10k_connection_lb")
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step9b_inputs(args: argparse.Namespace) -> dict[str, object]:
    """确认 Step 9A 已人工接受，并读取两组 10k 发丝及四视图输入。"""

    parent = json.loads(args.step9a_report.read_text(encoding="utf-8"))
    if not parent.get("passed_numeric_gates", False):
        raise RuntimeError("Step 9A 数值门禁未通过")
    if parent.get("human_review_status") != "accepted":
        raise RuntimeError("Step 9A 尚未被人工标记为 accepted")
    models = {
        "10k_baseline": read_ordered_strands(args.full_10k_baseline),
        "10k_connection_lb": read_ordered_strands(args.full_10k_candidate),
    }
    if any(len(strands) != 10000 for strands in models.values()):
        raise ValueError("Step 9B 的两组输入都必须严格包含 10000 根发束")
    images: dict[str, np.ndarray] = {}
    seg: dict[str, np.ndarray] = {}
    strand_maps: dict[str, np.ndarray] = {}
    for view in VIEWS:
        image_path = (
            args.front_image
            if view == "front"
            else args.data_root / "flux_redrawn" / f"{view}.png"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(args.data_root / "maps/seg" / f"{view}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        strand_map = cv2.imread(
            str(args.data_root / "maps/strand_map" / f"{view}.png"),
            cv2.IMREAD_COLOR,
        )
        if image is None or mask is None or strand_map is None:
            raise FileNotFoundError(f"缺少 {view} 图像、seg 或 strand_map")
        images[view] = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        seg[view] = cv2.resize(
            mask, (512, 512), interpolation=cv2.INTER_NEAREST
        ) > 127
        strand_maps[view] = cv2.resize(
            strand_map, (512, 512), interpolation=cv2.INTER_LINEAR
        )
    parting_mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if parting_mask is None:
        raise FileNotFoundError(f"缺少发缝 mask: {args.parting_mask}")
    parting_mask = cv2.resize(
        parting_mask, (512, 512), interpolation=cv2.INTER_NEAREST
    ) > 0
    head = o3d.io.read_triangle_mesh(str(args.head_mesh))
    if not head.has_vertices() or not head.has_triangles():
        raise ValueError("头模为空")
    return {
        "parent": parent,
        "models": models,
        "images": images,
        "seg": seg,
        "strand_maps": strand_maps,
        "parting_mask": parting_mask,
        "head": head,
    }


def compose_10k_comparison(
    output_dir: Path,
    images: dict[str, np.ndarray],
    rasters: dict[str, dict[str, dict[str, object]]],
    parting_mask: np.ndarray,
) -> dict[str, str]:
    """输出两组 10k 的四视图总表、正面发缝放大图和候选单视图。"""

    colors = {
        "10k_baseline": (255, 170, 40),
        "10k_connection_lb": (60, 240, 80),
    }
    rows = []
    individual: dict[str, str] = {}
    for view in VIEWS:
        panels = []
        for model in MODELS:
            base = images[view].copy()
            occupancy = np.asarray(rasters[model][view]["occupancy"], dtype=bool)
            overlay = base.copy()
            overlay[occupancy] = colors[model]
            panel = cv2.addWeighted(base, 0.58, overlay, 0.42, 0.0)
            cv2.putText(
                panel,
                f"{model} / {view}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
            if model == "10k_connection_lb":
                path = output_dir / f"10k_connection_lb_{view}_overlay.png"
                cv2.imwrite(str(path), panel)
                individual[view] = str(path)
        rows.append(np.concatenate(panels, axis=1))
    grid_path = output_dir / "10k_baseline_vs_connection_lb_multiview.png"
    cv2.imwrite(str(grid_path), np.concatenate(rows, axis=0))

    ys, xs = np.where(parting_mask)
    padding = 24
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, 512)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, 512)
    crops = []
    for model in MODELS:
        base = images["front"].copy()
        occupancy = np.asarray(rasters[model]["front"]["occupancy"], dtype=bool)
        overlay = base.copy()
        overlay[occupancy] = colors[model]
        crop = cv2.addWeighted(base, 0.58, overlay, 0.42, 0.0)[y0:y1, x0:x1]
        crop = cv2.resize(crop, (480, 512), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            crop,
            model,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        crops.append(crop)
    crop_path = output_dir / "10k_front_parting_crop_raw_comparison.png"
    cv2.imwrite(str(crop_path), np.concatenate(crops, axis=1))
    return {
        "multiview_grid": str(grid_path),
        "front_parting_crop": str(crop_path),
        **{
            f"candidate_{view}": path
            for view, path in individual.items()
        },
    }


def run_step9b_scale_gate(args: argparse.Namespace) -> dict[str, object]:
    """执行两组完整 10k 的公平物理采样四视图比较。"""

    started = time.perf_counter()
    config = VisibilityConfig()
    inputs = load_step9b_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = {view: load_view_camera(args, view) for view in VIEWS}
    head_buffers = {
        view: build_head_depth_map(
            inputs["head"],
            cameras[view]["calib"],
            cameras[view]["toward_camera"],
            config.image_size,
        )
        for view in VIEWS
    }
    rasters: dict[str, dict[str, dict[str, object]]] = {
        model: {} for model in MODELS
    }
    metrics: dict[str, dict[str, dict[str, object]]] = {
        model: {} for model in MODELS
    }
    timings: dict[str, dict[str, float]] = {model: {} for model in MODELS}
    for model in MODELS:
        for view in VIEWS:
            view_started = time.perf_counter()
            segments = collect_projected_segments(
                inputs["models"][model],
                cameras[view]["calib"],
                config.image_size,
                config.physical_sampling_m,
            )
            head_depth, origins, direction = head_buffers[view]
            raster = rasterize_hair_view(
                segments, head_depth, origins, direction, config
            )
            rasters[model][view] = raster
            metrics[model][view] = evaluate_model_view(
                segments,
                raster,
                inputs["seg"][view],
                inputs["strand_maps"][view],
                inputs["parting_mask"] if view == "front" else None,
            )
            timings[model][view] = float(time.perf_counter() - view_started)

    baseline = metrics["10k_baseline"]
    candidate = metrics["10k_connection_lb"]
    visibility_delta = (
        candidate["front"]["parting_region"]["scalp_visibility_ratio"]
        - baseline["front"]["parting_region"]["scalp_visibility_ratio"]
    )
    direction_delta = (
        candidate["front"]["direction_error_deg"]["q50"]
        - baseline["front"]["direction_error_deg"]["q50"]
    )
    silhouette_delta = {
        view: candidate[view]["silhouette_iou"]
        - baseline[view]["silhouette_iou"]
        for view in VIEWS
    }
    behind_delta = {
        view: candidate[view]["behind_head_fraction"]
        - baseline[view]["behind_head_fraction"]
        for view in VIEWS
    }
    penetration = inputs["parent"]["metrics"]["penetration_points"]
    baseline_size = args.full_10k_baseline.stat().st_size
    candidate_size = args.full_10k_candidate.stat().st_size
    size_ratio = float(candidate_size / max(baseline_size, 1))
    numeric_gates = {
        "both_inputs_exactly_10000_strands": all(
            len(inputs["models"][model]) == 10000 for model in MODELS
        ),
        "front_parting_scalp_visibility_not_worse": visibility_delta >= -1e-12,
        "front_direction_error_not_worse_over_2deg": (
            direction_delta <= config.direction_error_allowance_deg
        ),
        "all_view_silhouette_iou_not_worse_over_0p02": (
            min(silhouette_delta.values()) >= -config.silhouette_iou_allowance
        ),
        "all_view_behind_head_fraction_not_worse_over_0p005": (
            max(behind_delta.values())
            <= config.behind_head_fraction_allowance
        ),
        "no_new_surface_penetration": int(penetration["added"]) == 0,
        "candidate_file_size_not_over_1p2x_baseline": size_ratio <= 1.2,
    }
    outputs = compose_10k_comparison(
        args.output_dir,
        inputs["images"],
        rasters,
        inputs["parting_mask"],
    )
    report = {
        "image_id": args.image_id,
        "step": "step_09b_10k_render_gate",
        "inputs": {
            "full_10k_baseline": str(args.full_10k_baseline.resolve()),
            "full_10k_candidate": str(args.full_10k_candidate.resolve()),
            "front_image": str(args.front_image.resolve()),
            "front_background_policy": "raw_img.png only",
            "side_background_policy": "flux_redrawn view images; no Blender RGB used",
        },
        "config": config.__dict__,
        "metrics": {
            "models": metrics,
            "relative_to_10k_baseline": {
                "front_parting_scalp_visibility_delta": visibility_delta,
                "front_direction_error_q50_delta_deg": direction_delta,
                "silhouette_iou_delta": silhouette_delta,
                "behind_head_fraction_delta": behind_delta,
            },
            "surface_penetration": penetration,
            "performance": {
                "view_seconds": timings,
                "total_seconds": float(time.perf_counter() - started),
                "baseline_file_bytes": baseline_size,
                "candidate_file_bytes": candidate_size,
                "candidate_to_baseline_size_ratio": size_ratio,
            },
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": outputs,
        "parent_step_report": str(args.step9a_report.resolve()),
    }
    report_path = args.output_dir / "step_09b_10k_render_gate_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_step9b_arg_parser() -> argparse.ArgumentParser:
    """构造 Step 9B 参数。"""

    base = DEFAULT_ROOT / "pde_governance"
    step9a = base / "connection_lb_parting/step_09_10k_scale/step_09a_outer_coupling_repaired"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--step9a-report",
        type=Path,
        default=step9a / "step_09a_10k_outer_coupling_report.json",
    )
    parser.add_argument(
        "--full-10k-baseline",
        type=Path,
        default=base / "parting_topology_full_reconstruction_10k_128_20260828/hair_multiview.ply",
    )
    parser.add_argument(
        "--full-10k-candidate",
        type=Path,
        default=step9a / "full_10k_connection_lb.ply",
    )
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
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
    parser.add_argument(
        "--parting-mask",
        type=Path,
        default=base / "front_parting_dinov3_temp_rerun/parting_region_mask.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "connection_lb_parting/step_09_10k_scale/step_09b_render_gate",
    )
    return parser


def step9b_main() -> None:
    """运行 Step 9B 并打印门禁摘要。"""

    report = run_step9b_scale_gate(build_step9b_arg_parser().parse_args())
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
    step9b_main()
