#!/usr/bin/env python3
"""审计候选发缝曲线是否位于 front 相机可见的头模近表面。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_visibility_audit_inputs(args: argparse.Namespace) -> dict:
    """读取曲线、头模和原图标定，构造 HairStep 到原图 NDC 的矩阵。"""

    def build_calib(path: Path) -> np.ndarray:
        param = np.load(path, allow_pickle=True).item()
        center = np.asarray(param["center"], dtype=np.float64).reshape(3)
        rotation = np.asarray(param["R"], dtype=np.float64).reshape(3, 3)
        scale = float(np.asarray(param["scale"]).reshape(-1)[0])
        ortho_ratio = float(np.asarray(param["ortho_ratio"]).reshape(-1)[0])
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -(rotation @ center)
        intrinsic = np.eye(4, dtype=np.float64)
        intrinsic[0, 0] = scale / ortho_ratio / 512.0
        intrinsic[1, 1] = -scale / ortho_ratio / 512.0
        intrinsic[2, 2] = scale / ortho_ratio / 512.0
        return intrinsic @ extrinsic

    calib = build_calib(args.front_calib)
    if args.front_depth_calib is not None:
        depth_calib = build_calib(args.front_depth_calib)
        calib[2] = depth_calib[2]

    mesh = trimesh.load(str(args.head_mesh), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"无效头模: {args.head_mesh}")

    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    parting_mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取 front 图像: {args.front_image}")
    if parting_mask is None:
        raise FileNotFoundError(f"无法读取发缝 mask: {args.parting_mask}")
    if parting_mask.shape != image.shape[:2]:
        parting_mask = cv2.resize(
            parting_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    curves = {}
    for name, path in (("v24", args.v24_curve), ("v30", args.v30_curve)):
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError(f"{name} 曲线 points 形状无效: {points.shape}")
        curves[name] = {"path": Path(path), "points": points}

    return {
        "calib": calib,
        "mesh": mesh,
        "image": image,
        "parting_mask": parting_mask,
        "curves": curves,
    }


def project_curve_to_front_camera(
    points: np.ndarray, calib: np.ndarray, width: int, height: int
) -> dict:
    """将 HairStep 三维点投影到 front 图像，并构造对应的正向射线。"""
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    clip = homogeneous @ calib.T
    safe_w = np.abs(clip[:, 3]) > 1e-10
    ndc = np.full((len(points), 3), np.nan, dtype=np.float64)
    ndc[safe_w] = clip[safe_w, :3] / clip[safe_w, 3:4]
    pixels = np.column_stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1),
            (ndc[:, 1] + 1.0) * 0.5 * max(height - 1, 1),
        ]
    )
    inside = (
        safe_w
        & np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )

    inverse = np.linalg.inv(calib)
    ray_clip = np.column_stack(
        [ndc[:, :2], np.full(len(points), -1.0), np.ones(len(points))]
    )
    end_clip = ray_clip.copy()
    end_clip[:, 2] = 1.0
    ray_world = ray_clip @ inverse.T
    end_world = end_clip @ inverse.T
    ray_origins = ray_world[:, :3] / ray_world[:, 3:4]
    ray_ends = end_world[:, :3] / end_world[:, 3:4]
    ray_directions = ray_ends - ray_origins
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-12
    curve_depth = np.sum((points - ray_origins) * ray_directions, axis=1)
    return {
        "ndc": ndc,
        "pixels": pixels,
        "inside": inside,
        "ray_origins": ray_origins,
        "ray_directions": ray_directions,
        "curve_depth": curve_depth,
    }


def measure_curve_visibility(
    mesh: trimesh.Trimesh, points: np.ndarray, projection: dict
) -> dict:
    """对每条投影射线去重交点，并计算曲线点的深度排名。"""
    count = len(points)
    locations, ray_ids, triangle_ids = mesh.ray.intersects_location(
        projection["ray_origins"],
        projection["ray_directions"],
        multiple_hits=True,
    )
    has_hit = np.zeros(count, dtype=bool)
    nearest_depth = np.full(count, np.nan, dtype=np.float64)
    selected_depth = np.full(count, np.nan, dtype=np.float64)
    selected_rank = np.full(count, -1, dtype=np.int32)
    selected_facing = np.full(count, np.nan, dtype=np.float64)
    hit_count = np.zeros(count, dtype=np.int32)

    for ray_id in range(count):
        match = np.flatnonzero(ray_ids == ray_id)
        if len(match) == 0:
            continue
        distances = np.sum(
            (locations[match] - projection["ray_origins"][ray_id])
            * projection["ray_directions"][ray_id],
            axis=1,
        )
        positive = np.flatnonzero(distances >= -1e-7)
        if len(positive) == 0:
            continue
        order = positive[np.argsort(distances[positive])]
        unique = []
        for local_id in order:
            if not unique or abs(distances[local_id] - distances[unique[-1]]) > 1e-5:
                unique.append(int(local_id))
        unique = np.asarray(unique, dtype=np.int32)
        depths = distances[unique]
        chosen = int(np.argmin(np.abs(depths - projection["curve_depth"][ray_id])))
        chosen_match = match[unique[chosen]]
        normal = np.asarray(mesh.face_normals[triangle_ids[chosen_match]], dtype=np.float64)

        has_hit[ray_id] = True
        nearest_depth[ray_id] = depths[0]
        selected_depth[ray_id] = depths[chosen]
        selected_rank[ray_id] = chosen + 1
        selected_facing[ray_id] = np.dot(
            normal, -projection["ray_directions"][ray_id]
        )
        hit_count[ray_id] = len(depths)

    valid = projection["inside"] & has_hit
    nearest_error = np.abs(projection["curve_depth"] - nearest_depth)
    selected_error = np.abs(projection["curve_depth"] - selected_depth)
    far_side = valid & (selected_rank > 1)
    nearest_visible = valid & (selected_rank == 1)
    return {
        **projection,
        "has_hit": has_hit,
        "valid": valid,
        "nearest_depth": nearest_depth,
        "selected_depth": selected_depth,
        "selected_rank": selected_rank,
        "selected_facing": selected_facing,
        "hit_count": hit_count,
        "nearest_error": nearest_error,
        "selected_error": selected_error,
        "far_side": far_side,
        "nearest_visible": nearest_visible,
    }


def snap_curve_to_visible_surface(points: np.ndarray, audit: dict) -> np.ndarray:
    """保持二维投影轨迹不变，将曲线深度替换为每条射线的第一交点。"""
    if len(points) != len(audit["valid"]) or not np.all(audit["valid"]):
        invalid = np.flatnonzero(~audit["valid"])
        raise RuntimeError(
            f"v24 存在 {len(invalid)} 个无有效前表面交点，无法安全修正: "
            f"{invalid[:10].tolist()}"
        )
    corrected = (
        audit["ray_origins"]
        + audit["ray_directions"] * audit["nearest_depth"][:, None]
    )
    if not np.all(np.isfinite(corrected)):
        raise RuntimeError("可见表面修正产生了非有限坐标")
    return corrected


def build_visible_curve_from_mask(
    mesh: trimesh.Trimesh, mask: np.ndarray, calib: np.ndarray
) -> tuple[np.ndarray, dict]:
    """从原图 mask 逐行中心发射射线，并固定选择第一个头模交点。"""
    mask = np.asarray(mask) > 0
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 2:
        raise RuntimeError("正确发缝 mask 的有效行不足")
    raw_centers = np.asarray(
        [np.median(np.flatnonzero(mask[row])) for row in rows],
        dtype=np.float64,
    )
    centers = cv2.GaussianBlur(
        raw_centers[:, None], (1, 5), 0, borderType=cv2.BORDER_REPLICATE
    ).ravel()
    for index, row in enumerate(rows):
        columns = np.flatnonzero(mask[row])
        centers[index] = np.clip(centers[index], columns[0], columns[-1])

    ndc = np.column_stack(
        [
            centers / max(width - 1, 1) * 2.0 - 1.0,
            rows / max(height - 1, 1) * 2.0 - 1.0,
        ]
    )
    inverse = np.linalg.inv(calib)
    near_clip = np.column_stack(
        [ndc, np.full(len(rows), -1.0), np.ones(len(rows))]
    )
    far_clip = near_clip.copy()
    far_clip[:, 2] = 1.0
    near_world = near_clip @ inverse.T
    far_world = far_clip @ inverse.T
    ray_origins = near_world[:, :3] / near_world[:, 3:4]
    ray_ends = far_world[:, :3] / far_world[:, 3:4]
    ray_directions = ray_ends - ray_origins
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-12

    locations, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )
    points = np.full((len(rows), 3), np.nan, dtype=np.float64)
    for ray_id in range(len(rows)):
        hits = locations[ray_ids == ray_id]
        if len(hits) == 0:
            continue
        depths = np.sum(
            (hits - ray_origins[ray_id]) * ray_directions[ray_id], axis=1
        )
        positive = np.flatnonzero(depths >= -1e-7)
        if len(positive):
            chosen = positive[np.argmin(depths[positive])]
            points[ray_id] = hits[chosen]
    valid = np.all(np.isfinite(points), axis=1)
    hit_coverage = float(np.mean(valid))
    if hit_coverage < 0.95:
        missing = rows[~valid]
        raise RuntimeError(
            f"原图发缝只有 {hit_coverage:.1%} 的行命中头模第一表面；"
            f"未命中 {len(missing)} 行: "
            f"{missing[:10].tolist()}"
        )
    return points[valid], {
        "rows": rows[valid],
        "raw_centers": raw_centers[valid],
        "smoothed_centers": centers[valid],
        "missing_rows": rows[~valid],
        "ray_hit_row_coverage": hit_coverage,
        "center_smoothing_px_q95": float(
            np.quantile(np.abs(centers[valid] - raw_centers[valid]), 0.95)
        ),
    }


def measure_curve_image_alignment(audit: dict, mask: np.ndarray) -> dict:
    """独立测量曲线投影与原图发缝中心线、区域及行区间的一致性。"""
    mask = np.asarray(mask) > 0
    height, width = mask.shape
    mask_rows = np.flatnonzero(mask.any(axis=1))
    center_by_row = np.full(height, np.nan, dtype=np.float64)
    for row in mask_rows:
        center_by_row[row] = float(np.median(np.flatnonzero(mask[row])))

    pixels = np.asarray(audit["pixels"], dtype=np.float64)
    px = np.rint(pixels[:, 0]).astype(np.int64)
    py = np.rint(pixels[:, 1]).astype(np.int64)
    inside_image = (
        np.all(np.isfinite(pixels), axis=1)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    on_mask_row = inside_image.copy()
    on_mask_row[inside_image] &= np.isfinite(center_by_row[py[inside_image]])
    center_errors = np.abs(
        pixels[on_mask_row, 0] - center_by_row[py[on_mask_row]]
    )
    inside_mask = np.zeros(len(pixels), dtype=bool)
    inside_mask[inside_image] = mask[py[inside_image], px[inside_image]]
    covered_rows = np.unique(py[on_mask_row])
    row_coverage = float(len(covered_rows) / max(len(mask_rows), 1))
    return {
        "centerline_error_px": {
            "q50": float(np.quantile(center_errors, 0.50)) if len(center_errors) else None,
            "q95": float(np.quantile(center_errors, 0.95)) if len(center_errors) else None,
            "max": float(np.max(center_errors)) if len(center_errors) else None,
        },
        "inside_mask_count": int(np.sum(inside_mask)),
        "inside_mask_ratio": float(np.mean(inside_mask)),
        "covered_mask_rows": int(len(covered_rows)),
        "total_mask_rows": int(len(mask_rows)),
        "mask_row_coverage": row_coverage,
    }


def render_curve_visibility_overlay(
    output_path: Path,
    image: np.ndarray,
    parting_mask: np.ndarray,
    audit: dict,
    label: str,
) -> None:
    """绘制 mask、曲线以及最近/远侧命中分类。"""
    canvas = image.copy()
    mask = parting_mask > 127
    magenta = np.zeros_like(canvas)
    magenta[:, :, :] = (180, 40, 220)
    canvas[mask] = cv2.addWeighted(canvas[mask], 0.55, magenta[mask], 0.45, 0)
    pixels = audit["pixels"]

    for index in range(1, len(pixels)):
        if audit["inside"][index - 1] and audit["inside"][index]:
            p0 = tuple(np.rint(pixels[index - 1]).astype(int))
            p1 = tuple(np.rint(pixels[index]).astype(int))
            cv2.line(canvas, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)

    for index, pixel in enumerate(pixels):
        if not audit["inside"][index]:
            continue
        center = tuple(np.rint(pixel).astype(int))
        cv2.circle(canvas, center, 5, (255, 170, 0), 1, cv2.LINE_AA)
        if not audit["has_hit"][index]:
            color = (0, 165, 255)
        elif audit["far_side"][index]:
            color = (0, 0, 255)
        else:
            color = (40, 220, 40)
        cv2.circle(canvas, center, 2, color, -1, cv2.LINE_AA)

    valid = audit["valid"]
    q95_mm = (
        float(np.quantile(audit["nearest_error"][valid], 0.95) * 1000.0)
        if np.any(valid)
        else float("nan")
    )
    lines = [
        f"{label}: green=nearest, red=far side, orange=no hit",
        f"valid={int(np.sum(valid))}/{len(valid)}  far_side={int(np.sum(audit['far_side']))}",
        f"nearest depth error q95={q95_mm:.3f} mm",
        "magenta=2D parting mask; blue ring=nearest head hit pixel",
    ]
    for row, line in enumerate(lines):
        y = 22 + row * 20
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"无法写入图像: {output_path}")


def render_curve_depth_comparison(output_path: Path, audits: dict) -> None:
    """绘制原始曲线与可见表面修正版的误差、排名和射线深度。"""
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    colors = {
        "v24": "#1b9e77",
        "v30": "#d95f02",
        "raw_mask_raycast": "#7570b3",
    }
    for name, audit in audits.items():
        sample = np.arange(len(audit["curve_depth"]))
        valid = audit["valid"]
        axes[0].plot(sample[valid], audit["nearest_error"][valid] * 1000.0, ".-", label=name, color=colors[name])
        axes[1].step(sample[valid], audit["selected_rank"][valid], where="mid", label=name, color=colors[name])
        axes[2].plot(sample[valid], audit["curve_depth"][valid], "-", label=f"{name} curve", color=colors[name])
        axes[2].plot(sample[valid], audit["nearest_depth"][valid], "--", label=f"{name} nearest", color=colors[name], alpha=0.7)
    axes[0].axhline(1.0, color="black", linestyle=":", label="1 mm gate")
    axes[0].set_ylabel("nearest depth error (mm)")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("closest surface depth rank")
    axes[1].set_yticks(sorted({1, *[int(v) for a in audits.values() for v in a["selected_rank"] if v > 0]}))
    axes[2].set_ylabel("ray depth (HairStep unit)")
    axes[2].set_xlabel("curve sample index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle("Raw-photo calibration audit: v24/v30 and mask raycast")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_parting_curve_visibility_audit(args: argparse.Namespace) -> dict:
    """运行 Step 1 审计，保存图像、机器报告和待人工确认的推荐曲线。"""
    inputs = load_visibility_audit_inputs(args)
    height, width = inputs["image"].shape[:2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = args.output_dir / "raw_parting_visible_surface_curve.npz"
    candidate_points, construction = build_visible_curve_from_mask(
        inputs["mesh"], inputs["parting_mask"], inputs["calib"]
    )
    np.savez_compressed(
        candidate_path,
        points=candidate_points.astype(np.float32),
        source_name=np.asarray("raw_mask_raycast"),
        source_mask=np.asarray(str(args.parting_mask)),
        source_calibration=np.asarray(str(args.front_calib)),
        construction=np.asarray("raw_mask_centerline_first_head_intersection"),
    )
    inputs["curves"]["raw_mask_raycast"] = {
        "path": candidate_path,
        "points": candidate_points,
    }

    audits = {}
    metrics = {}
    for name, curve in inputs["curves"].items():
        projection = project_curve_to_front_camera(
            curve["points"], inputs["calib"], width, height
        )
        audit = measure_curve_visibility(inputs["mesh"], curve["points"], projection)
        audits[name] = audit
        valid = audit["valid"]
        errors = audit["nearest_error"][valid]
        facing = audit["selected_facing"][valid]
        coverage = float(np.mean(valid))
        q95_mm = float(np.quantile(errors, 0.95) * 1000.0) if len(errors) else None
        image_alignment = measure_curve_image_alignment(
            audit, inputs["parting_mask"]
        )
        rank_values, rank_counts = np.unique(audit["selected_rank"][valid], return_counts=True)
        numeric_gates = {
            "far_side_hit_count_eq_0": int(np.sum(audit["far_side"])) == 0,
            "nearest_depth_error_q95_le_1mm": q95_mm is not None and q95_mm <= 1.0,
            "valid_projection_coverage_ge_0_95": coverage >= 0.95,
            "centerline_error_q95_le_1px": (
                image_alignment["centerline_error_px"]["q95"] is not None
                and image_alignment["centerline_error_px"]["q95"] <= 1.0
            ),
            "inside_mask_ratio_ge_0_95": image_alignment["inside_mask_ratio"] >= 0.95,
            "mask_row_coverage_ge_0_95": image_alignment["mask_row_coverage"] >= 0.95,
        }
        metrics[name] = {
            "point_count": int(len(valid)),
            "valid_projection_count": int(np.sum(valid)),
            "valid_projection_coverage": coverage,
            "out_of_frame_count": int(np.sum(~audit["inside"])),
            "no_mesh_hit_count": int(np.sum(audit["inside"] & ~audit["has_hit"])),
            "nearest_visible_count": int(np.sum(audit["nearest_visible"])),
            "far_side_hit_count": int(np.sum(audit["far_side"])),
            "nearest_depth_error_mm": {
                "q50": float(np.quantile(errors, 0.50) * 1000.0) if len(errors) else None,
                "q95": q95_mm,
                "max": float(np.max(errors) * 1000.0) if len(errors) else None,
            },
            "selected_surface_error_mm_q95": float(np.quantile(audit["selected_error"][valid], 0.95) * 1000.0) if np.any(valid) else None,
            "front_facing_ratio": float(np.mean(facing > 0.0)) if len(facing) else None,
            "depth_rank_histogram": {str(int(k)): int(v) for k, v in zip(rank_values, rank_counts)},
            "image_alignment": image_alignment,
            "numeric_gates": numeric_gates,
            "passed_numeric_gates": bool(all(numeric_gates.values())),
        }

    for name, audit in audits.items():
        render_curve_visibility_overlay(
            args.output_dir / f"{name}_curve_front_overlay.png",
            inputs["image"],
            inputs["parting_mask"],
            audit,
            name,
        )
    render_curve_depth_comparison(
        args.output_dir / "curve_depth_comparison.png", audits
    )

    ranked = sorted(
        metrics,
        key=lambda name: (
            not metrics[name]["passed_numeric_gates"],
            metrics[name]["image_alignment"]["centerline_error_px"]["q95"]
            if metrics[name]["image_alignment"]["centerline_error_px"]["q95"] is not None
            else float("inf"),
            metrics[name]["far_side_hit_count"],
            metrics[name]["nearest_depth_error_mm"]["q95"]
            if metrics[name]["nearest_depth_error_mm"]["q95"] is not None
            else float("inf"),
            -metrics[name]["valid_projection_coverage"],
        ),
    )
    recommended = ranked[0]
    accepted_path = args.output_dir / "accepted_curve_candidate.npz"
    np.savez_compressed(
        accepted_path,
        points=inputs["curves"][recommended]["points"].astype(np.float32),
        source_name=np.asarray(recommended),
        source_path=np.asarray(str(inputs["curves"][recommended]["path"])),
        numeric_gates_passed=np.asarray(metrics[recommended]["passed_numeric_gates"]),
        human_review_status=np.asarray("pending"),
    )

    report = {
        "image_id": args.image_id,
        "step": "step_01_visible_curve",
        "purpose": "审计候选发缝曲线是否位于 front 可见近表面；本步不求解新方向场",
        "inputs": {
            "head_mesh": str(args.head_mesh),
            "front_calib": str(args.front_calib),
            "front_depth_calib": str(args.front_depth_calib),
            "front_image": str(args.front_image),
            "parting_mask": str(args.parting_mask),
            "v24_curve": str(args.v24_curve),
            "v30_curve": str(args.v30_curve),
        },
        "config": {
            "calibration_coordinate_system": "raw_photo",
            "calibration_load_size": 1024,
            "surface_hit_dedup_tolerance": 1e-5,
            "far_side_definition": "与曲线点深度最近的去重交点排名大于 1",
            "candidate_construction": "正确 mask 逐行中心轻微平滑后，沿原图标定射线选择第一头模交点",
            "center_smoothing_px_q95": construction["center_smoothing_px_q95"],
            "ray_hit_row_coverage": construction["ray_hit_row_coverage"],
            "missing_mask_rows": construction["missing_rows"].astype(int).tolist(),
            "nearest_surface_gate_mm": 1.0,
            "minimum_valid_projection_coverage": 0.95,
            "maximum_centerline_error_q95_px": 1.0,
            "minimum_inside_mask_ratio": 0.95,
            "minimum_mask_row_coverage": 0.95,
        },
        "metrics": metrics,
        "recommended_curve": recommended,
        "recommendation_basis": "同时通过原图二维对齐与三维可见近表面门禁",
        "passed_numeric_gates": metrics[recommended]["passed_numeric_gates"],
        "requires_human_review": True,
        "human_review_status": "pending",
        "parent_step_report": "docs/PARTING_CONNECTION_LAPLACE_BELTRAMI_PLAN.md#step-0方案与实验协议锁定",
        "outputs": {
            "v24_overlay": str(args.output_dir / "v24_curve_front_overlay.png"),
            "v30_overlay": str(args.output_dir / "v30_curve_front_overlay.png"),
            "raw_mask_raycast_overlay": str(args.output_dir / "raw_mask_raycast_curve_front_overlay.png"),
            "raw_mask_raycast_curve": str(candidate_path),
            "depth_comparison": str(args.output_dir / "curve_depth_comparison.png"),
            "candidate": str(accepted_path),
        },
    }
    report_path = args.output_dir / "curve_visibility_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--parting-mask", type=Path, default=DEFAULT_ROOT / "pde_governance/front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--v24-curve", type=Path, default=DEFAULT_ROOT / "pde_governance/scalp_geodesic_reference_v24/front_parting_scalp_curve.npz")
    parser.add_argument("--v30-curve", type=Path, default=DEFAULT_ROOT / "pde_governance/pde_parting_hybrid_cap_v30/front_parting_scalp_curve.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_01_visible_curve")
    parsed_args = parser.parse_args()
    result = run_parting_curve_visibility_audit(parsed_args)
    print(json.dumps({"recommended_curve": result["recommended_curve"], "passed_numeric_gates": result["passed_numeric_gates"], "human_review_status": result["human_review_status"], "report": str(parsed_args.output_dir / "curve_visibility_report.json")}, ensure_ascii=False, indent=2))
