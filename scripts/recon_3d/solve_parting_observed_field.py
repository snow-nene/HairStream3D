#!/usr/bin/env python3
"""Step 5：在 beta=60° 两岸硬边界上加入 front strand map 高置信度观测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.recon_3d.solve_parting_bank_only_field import (
    build_bank_boundary_values,
    calibration_from_param,
    match_bank_vertices,
    project_points_to_image,
    recover_line_tangents,
)
from scripts.recon_3d.validate_parting_connection_lb import (
    cotangent_connection_operator,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step5_inputs(args: argparse.Namespace) -> dict:
    """验证 Step 4 人工选择，并读取 beta=60° 基线、front 观测和双 atlas。"""
    step2_report = json.loads(args.step2_report.read_text(encoding="utf-8"))
    step4_report = json.loads(args.step4_report.read_text(encoding="utf-8"))
    selection = json.loads(args.bank_selection.read_text(encoding="utf-8"))
    if not step2_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 2 数值门禁未通过")
    if not step4_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 4 数值门禁未通过")
    if step4_report.get("human_review_status") != "approved":
        raise RuntimeError("Step 4 尚未人工验收")
    selected_beta = float(selection["selected_beta_deg"])
    if abs(selected_beta - 60.0) > 1e-9 or not selection.get("confirmed_by_user", False):
        raise RuntimeError("Step 5 要求用户已确认 beta=60°")

    meshes = {}
    for chart_id, path in (("left", args.left_atlas), ("right", args.right_atlas)):
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"{chart_id} atlas 无效")
        meshes[chart_id] = mesh
    cap = trimesh.load(str(args.cap_mesh), process=False)
    if isinstance(cap, trimesh.Scene):
        cap = trimesh.util.concatenate(tuple(cap.geometry.values()))

    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
        shores = {
            "left": np.asarray(archive["left_shore_points"], dtype=np.float64),
            "right": np.asarray(archive["right_shore_points"], dtype=np.float64),
        }
    with np.load(args.bank_fields, allow_pickle=False) as archive:
        baseline_q = {
            "left": np.asarray(archive[selection["field_keys"]["left"]], dtype=np.complex128),
            "right": np.asarray(archive[selection["field_keys"]["right"]], dtype=np.complex128),
        }

    strand_bgr = cv2.imread(str(args.front_strand), cv2.IMREAD_COLOR)
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(args.front_seg), cv2.IMREAD_GRAYSCALE)
    if strand_bgr is None or image is None or seg is None:
        raise FileNotFoundError("无法读取 front strand map、raw_img 或 seg")
    strand_rgb = cv2.cvtColor(strand_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    if strand_rgb.shape[:2] != image.shape[:2] or seg.shape != image.shape[:2]:
        raise ValueError("front strand map、raw_img 与 seg 尺寸不一致")
    calib = calibration_from_param(args.front_calib)
    depth_calib = calibration_from_param(args.front_depth_calib)
    calib[2] = depth_calib[2]
    return {
        "step2_report": step2_report,
        "step4_report": step4_report,
        "selection": selection,
        "meshes": meshes,
        "cap": cap,
        "curve": curve,
        "shores": shores,
        "baseline_q": baseline_q,
        "strand_rgb": strand_rgb,
        "image": image,
        "seg": seg > 127,
        "calib": calib,
    }


def compute_strand_confidence_map(
    strand_rgb: np.ndarray, seg: np.ndarray
) -> dict[str, np.ndarray]:
    """从局部二重角一致性、seg 边界距离和编码有效性构造 front 置信度。"""
    hair = (strand_rgb[:, :, 0] > 0.9) & np.asarray(seg, dtype=bool)
    dx = 1.0 - 2.0 * strand_rgb[:, :, 2]
    dy = 2.0 * strand_rgb[:, :, 1] - 1.0
    magnitude = np.sqrt(dx * dx + dy * dy)
    valid = hair & (magnitude > 0.25)
    dx = np.where(valid, dx / np.maximum(magnitude, 1e-12), 0.0)
    dy = np.where(valid, dy / np.maximum(magnitude, 1e-12), 0.0)
    q_image = (dx + 1j * dy) ** 2

    mask_float = valid.astype(np.float64)
    denominator = cv2.GaussianBlur(mask_float, (0, 0), 3.0) + 1e-8
    mean_real = cv2.GaussianBlur(mask_float * q_image.real, (0, 0), 3.0) / denominator
    mean_imag = cv2.GaussianBlur(mask_float * q_image.imag, (0, 0), 3.0) / denominator
    coherence = np.clip(np.sqrt(mean_real * mean_real + mean_imag * mean_imag), 0.0, 1.0)
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    boundary_confidence = np.clip((distance - 2.0) / 10.0, 0.0, 1.0)
    encoding_confidence = np.clip((magnitude - 0.25) / 0.65, 0.0, 1.0)
    confidence = coherence * boundary_confidence * encoding_confidence * mask_float
    return {
        "dx": dx,
        "dy": dy,
        "q_image": q_image,
        "coherence": coherence,
        "boundary_distance_px": distance,
        "confidence": confidence,
        "valid": valid,
    }


def sample_image_bilinear(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """在浮点像素位置双线性采样单通道图。"""
    sample_x = pixels[:, 0].astype(np.float32).reshape(-1, 1)
    sample_y = pixels[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(
        np.asarray(image, dtype=np.float64),
        sample_x,
        sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).reshape(-1)


def project_front_observations_to_atlas(
    mesh: trimesh.Trimesh,
    operator: dict,
    strand: dict[str, np.ndarray],
    calib: np.ndarray,
    image_shape: tuple[int, ...],
    confidence_threshold: float,
) -> dict:
    """把 front 二维线方向经投影 Jacobian 提升并投影到头皮切平面。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    pixels = project_points_to_image(vertices, calib, image_shape)
    height, width = image_shape[:2]
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )
    dx = sample_image_bilinear(strand["dx"], pixels)
    dy = sample_image_bilinear(strand["dy"], pixels)
    image_confidence = sample_image_bilinear(strand["confidence"], pixels)

    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    clip = homogeneous @ calib.T
    uv = clip[:, :2] / np.maximum(np.abs(clip[:, 3:4]), 1e-15)
    depth = clip[:, 2] / np.maximum(np.abs(clip[:, 3]), 1e-15)
    inverse = np.linalg.inv(calib)

    def unproject(local_uv: np.ndarray) -> np.ndarray:
        local_clip = np.column_stack([local_uv, depth, np.ones(len(local_uv))])
        world = local_clip @ inverse.T
        return world[:, :3] / np.maximum(np.abs(world[:, 3:4]), 1e-15)

    epsilon = 2.0 / max(height, width)
    base = unproject(uv)
    tangent_u = unproject(uv + np.array([epsilon, 0.0])) - base
    tangent_v = unproject(uv + np.array([0.0, epsilon])) - base
    normals = operator["frames"][2]
    center_uv = np.zeros((1, 2), dtype=np.float64)
    near_clip = np.column_stack([center_uv, np.array([1.5]), np.ones(1)])
    far_clip = np.column_stack([center_uv, np.array([-0.5]), np.ones(1)])
    near_world = near_clip @ inverse.T
    far_world = far_clip @ inverse.T
    ray = far_world[0, :3] / far_world[0, 3] - near_world[0, :3] / near_world[0, 3]
    ray /= np.linalg.norm(ray)
    image_plane_direction = dx[:, None] * tangent_u + dy[:, None] * tangent_v
    normal_ray = normals @ ray
    depth_component = -np.sum(image_plane_direction * normals, axis=1) / np.where(
        np.abs(normal_ray) > 1e-10, normal_ray, np.inf
    )
    direction = image_plane_direction + depth_component[:, None] * ray
    direction_norm = np.linalg.norm(direction, axis=1)
    direction /= np.maximum(direction_norm[:, None], 1e-15)

    facing_raw = normals @ (-ray)
    facing_confidence = np.clip((facing_raw - 0.05) / 0.45, 0.0, 1.0)
    confidence = image_confidence * facing_confidence
    valid = inside & (direction_norm > 1e-10) & (confidence >= float(confidence_threshold))
    confidence = np.where(valid, confidence, 0.0)
    e1, e2, _ = operator["frames"]
    local = np.sum(direction * e1, axis=1) + 1j * np.sum(direction * e2, axis=1)
    q_obs = local * local
    q_obs[~valid] = 0.0
    return {
        "pixels": pixels,
        "q_obs": q_obs,
        "direction_world": direction,
        "confidence": confidence,
        "valid": valid,
        "front_facing": facing_raw > 0.05,
        "facing_raw": facing_raw,
        "image_dx": dx,
        "image_dy": dy,
    }


def solve_screened_connection_field(
    operator: dict,
    q_obs: np.ndarray,
    confidence: np.ndarray,
    bank_ids: np.ndarray,
    bank_q: np.ndarray,
    screening_length_m: float,
) -> tuple[np.ndarray, dict]:
    """以硬岸线 Dirichlet 和面积加权观测项求解 screened connection 场。"""
    screen_weight = 1.0 / float(screening_length_m) ** 2
    observation_diagonal = operator["mass"] * confidence * screen_weight
    system = operator["stiffness"] + sparse.diags(observation_diagonal)
    rhs = observation_diagonal * q_obs
    fixed = np.zeros(system.shape[0], dtype=bool)
    fixed[bank_ids] = True
    free_ids = np.flatnonzero(~fixed)
    solution = np.zeros(system.shape[0], dtype=np.complex128)
    solution[bank_ids] = bank_q
    lhs = system[free_ids][:, free_ids]
    reduced_rhs = rhs[free_ids] - system[free_ids][:, bank_ids] @ bank_q
    solution[free_ids] = spsolve(lhs, reduced_rhs)
    equation_residual = np.linalg.norm(lhs @ solution[free_ids] - reduced_rhs) / max(
        np.linalg.norm(reduced_rhs), 1e-15
    )
    return solution, {
        "screening_length_m": float(screening_length_m),
        "screen_weight_per_m2": float(screen_weight),
        "observation_vertex_count": int(np.count_nonzero(confidence > 0.0)),
        "equation_relative_residual": float(equation_residual),
        "bank_dirichlet_max_residual": float(
            np.max(np.abs(solution[bank_ids] - bank_q), initial=0.0)
        ),
        "minimum_magnitude": float(np.min(np.abs(solution))),
        "q05_magnitude": float(np.quantile(np.abs(solution), 0.05)),
    }


def projected_line_error_deg(
    q: np.ndarray,
    frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    calib: np.ndarray,
    observation: dict,
) -> np.ndarray:
    """计算场投影到 front 后与 strand map 无符号方向的夹角。"""
    tangent = recover_line_tangents(
        q / np.maximum(np.abs(q), 1e-15), frames
    )
    projected = tangent @ calib[:2, :3].T
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-15)
    observed = np.column_stack([observation["image_dx"], observation["image_dy"]])
    observed /= np.maximum(np.linalg.norm(observed, axis=1, keepdims=True), 1e-15)
    cosine = np.clip(np.abs(np.sum(projected * observed, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def evaluate_observation_weight_candidate(
    screening_length_m: float,
    chart_data: dict[str, dict],
    calib: np.ndarray,
) -> dict:
    """求解一个 screening length，并统计相对 beta=60° 基线的 front 误差。"""
    charts = {}
    combined_baseline = []
    combined_observed = []
    for chart_id, data in chart_data.items():
        q, solve_metrics = solve_screened_connection_field(
            data["operator"],
            data["observation"]["q_obs"],
            data["observation"]["confidence"],
            data["bank_ids"],
            data["bank_q"],
            screening_length_m,
        )
        baseline_error = projected_line_error_deg(
            data["baseline_q"], data["operator"]["frames"], calib, data["observation"]
        )
        observed_error = projected_line_error_deg(
            q, data["operator"]["frames"], calib, data["observation"]
        )
        valid = data["observation"]["valid"]
        combined_baseline.append(baseline_error[valid])
        combined_observed.append(observed_error[valid])
        charts[chart_id] = {
            "q": q,
            "baseline_error_deg": baseline_error,
            "observed_error_deg": observed_error,
            "metrics": {
                "solve": solve_metrics,
                "baseline_error_median_deg": float(np.median(baseline_error[valid])),
                "observed_error_median_deg": float(np.median(observed_error[valid])),
                "observed_error_q95_deg": float(np.quantile(observed_error[valid], 0.95)),
            },
        }
    baseline_all = np.concatenate(combined_baseline)
    observed_all = np.concatenate(combined_observed)
    return {
        "screening_length_m": float(screening_length_m),
        "charts": charts,
        "metrics": {
            "baseline_error_median_deg": float(np.median(baseline_all)),
            "observed_error_median_deg": float(np.median(observed_all)),
            "observed_error_q95_deg": float(np.quantile(observed_all, 0.95)),
            "median_improvement_deg": float(np.median(baseline_all) - np.median(observed_all)),
        },
    }


def render_observed_field_front_preview(
    path: Path,
    image: np.ndarray,
    calib: np.ndarray,
    chart_data: dict[str, dict],
    selected: dict,
) -> None:
    """在 raw_img 上比较 beta=60° 基线、观测场和观测误差。"""
    panels = [image.copy(), image.copy(), image.copy()]
    colors = {"left": (255, 170, 20), "right": (20, 155, 255)}
    for chart_id, data in chart_data.items():
        vertices = np.asarray(data["mesh"].vertices)
        visible_ids = np.flatnonzero(data["observation"]["valid"])
        sample = visible_ids[:: max(1, len(visible_ids) // 260)]
        for panel_id, q in (
            (0, data["baseline_q"]),
            (1, selected["charts"][chart_id]["q"]),
        ):
            tangent = recover_line_tangents(
                q / np.maximum(np.abs(q), 1e-15), data["operator"]["frames"]
            )
            scale = 0.0032
            first = project_points_to_image(vertices[sample] - scale * tangent[sample], calib, image.shape)
            second = project_points_to_image(vertices[sample] + scale * tangent[sample], calib, image.shape)
            for start, stop in zip(first, second):
                cv2.line(
                    panels[panel_id], tuple(np.rint(start).astype(int)), tuple(np.rint(stop).astype(int)),
                    colors[chart_id], 1, cv2.LINE_AA,
                )
        errors = selected["charts"][chart_id]["observed_error_deg"]
        pixels = data["observation"]["pixels"]
        for vertex_id in sample:
            error = float(errors[vertex_id])
            ratio = np.clip(error / 45.0, 0.0, 1.0)
            color = (int(255 * ratio), int(255 * (1.0 - ratio)), 30)
            cv2.circle(
                panels[2], tuple(np.rint(pixels[vertex_id]).astype(int)), 2, color, -1, cv2.LINE_AA
            )
    labels = [
        "raw_img.png  beta=60 bank-only baseline",
        f"raw_img.png  + front observation  ell={selected['screening_length_m'] * 1000:g}mm",
        "front observation error: green=low, blue=high",
    ]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise RuntimeError(f"无法写出正面预览: {path}")


def render_observed_field_3d_preview(
    path: Path,
    chart_data: dict[str, dict],
    selected: dict,
    cap: trimesh.Trimesh,
) -> None:
    """绘制观测引导场和 front 置信度在双 atlas 上的分布。"""
    figure = plt.figure(figsize=(12, 6), dpi=180)
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    axes[0].set_title("selected observed connection field")
    axes[1].set_title("front observation confidence")
    cap_vertices = np.asarray(cap.vertices)
    cap_faces = np.asarray(cap.faces)
    all_vertices = np.concatenate([np.asarray(data["mesh"].vertices) for data in chart_data.values()])
    center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
    radius = 0.55 * float(np.max(np.ptp(all_vertices, axis=0)))
    confidence_scatter = None
    for axis in axes:
        axis.set_axis_off()
        axis.view_init(elev=20, azim=-82)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.add_collection3d(
            Poly3DCollection(cap_vertices[cap_faces], facecolor="#ee3dad", alpha=0.5, edgecolor="none")
        )
    colors = {"left": "#20a4f3", "right": "#ff9f1c"}
    for chart_id, data in chart_data.items():
        vertices = np.asarray(data["mesh"].vertices)
        faces = np.asarray(data["mesh"].faces)
        axes[0].add_collection3d(
            Poly3DCollection(vertices[faces], facecolor=colors[chart_id], alpha=0.07, edgecolor="none")
        )
        q = selected["charts"][chart_id]["q"]
        tangent = recover_line_tangents(q / np.maximum(np.abs(q), 1e-15), data["operator"]["frames"])
        sample = np.arange(0, len(vertices), max(1, len(vertices) // 230))
        scale = 0.0032
        for first, second in zip(vertices[sample] - scale * tangent[sample], vertices[sample] + scale * tangent[sample]):
            axes[0].plot(*np.stack([first, second]).T, color=colors[chart_id], linewidth=0.65)
        confidence_scatter = axes[1].scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            c=data["observation"]["confidence"], cmap="viridis", vmin=0.0, vmax=1.0, s=3,
        )
    figure.colorbar(confidence_scatter, ax=axes[1], shrink=0.62, pad=0.02, label="front confidence")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_observed_field_step(args: argparse.Namespace) -> dict:
    """运行 front 观测权重搜索、门禁、保存和人工检查预览。"""
    inputs = load_step5_inputs(args)
    strand = compute_strand_confidence_map(inputs["strand_rgb"], inputs["seg"])
    chart_data = {}
    for chart_id, mesh in inputs["meshes"].items():
        operator = cotangent_connection_operator(mesh)
        bank_ids, bank_match = match_bank_vertices(mesh, inputs["shores"][chart_id])
        bank_q, _ = build_bank_boundary_values(
            mesh, operator["frames"], bank_ids, inputs["curve"], 60.0
        )
        observation = project_front_observations_to_atlas(
            mesh,
            operator,
            strand,
            inputs["calib"],
            inputs["image"].shape,
            args.confidence_threshold,
        )
        chart_data[chart_id] = {
            "mesh": mesh,
            "operator": operator,
            "bank_ids": bank_ids,
            "bank_q": bank_q,
            "bank_match": bank_match,
            "observation": observation,
            "baseline_q": inputs["baseline_q"][chart_id],
        }
    candidates = [
        evaluate_observation_weight_candidate(length_mm / 1000.0, chart_data, inputs["calib"])
        for length_mm in args.screening_length_mm
    ]
    selected = min(
        candidates,
        key=lambda item: (
            item["metrics"]["observed_error_median_deg"],
            item["metrics"]["observed_error_q95_deg"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_observed_field_front_preview(
        args.output_dir / "observed_field_front_raw_comparison.png",
        inputs["image"],
        inputs["calib"],
        chart_data,
        selected,
    )
    render_observed_field_3d_preview(
        args.output_dir / "observed_field_3d_preview.png",
        chart_data,
        selected,
        inputs["cap"],
    )
    np.savez_compressed(
        args.output_dir / "observed_fields.npz",
        left_q=selected["charts"]["left"]["q"].astype(np.complex64),
        right_q=selected["charts"]["right"]["q"].astype(np.complex64),
        left_observation_confidence=chart_data["left"]["observation"]["confidence"].astype(np.float32),
        right_observation_confidence=chart_data["right"]["observation"]["confidence"].astype(np.float32),
        left_observation_q=chart_data["left"]["observation"]["q_obs"].astype(np.complex64),
        right_observation_q=chart_data["right"]["observation"]["q_obs"].astype(np.complex64),
        screening_length_m=np.asarray(selected["screening_length_m"]),
    )

    selected_metrics = selected["metrics"]
    max_bank_residual = max(
        chart["metrics"]["solve"]["bank_dirichlet_max_residual"]
        for chart in selected["charts"].values()
    )
    observation_counts = {
        chart_id: int(np.count_nonzero(data["observation"]["valid"]))
        for chart_id, data in chart_data.items()
    }
    hidden_max_confidence = max(
        float(np.max(data["observation"]["confidence"][~data["observation"]["front_facing"]], initial=0.0))
        for data in chart_data.values()
    )
    numeric_gates = {
        "front_median_error_improves_over_bank_only": selected_metrics["median_improvement_deg"] > 0.0,
        "bank_dirichlet_residual_le_1e-8": max_bank_residual <= 1e-8,
        "hidden_region_confidence_eq_0": hidden_max_confidence == 0.0,
        "both_charts_have_high_confidence_observations": min(observation_counts.values()) >= 20,
        "cross_atlas_coupling_outside_cap_eq_0": inputs["step2_report"]["cross_atlas_adjacency_outside_cap"] == 0,
    }
    candidate_reports = []
    for candidate in candidates:
        candidate_reports.append(
            {
                "screening_length_mm": candidate["screening_length_m"] * 1000.0,
                "metrics": candidate["metrics"],
                "charts": {
                    chart_id: chart["metrics"] for chart_id, chart in candidate["charts"].items()
                },
            }
        )
    report = {
        "image_id": args.image_id,
        "step": "step_05_observed_field",
        "purpose": "在 beta=60° 两岸硬边界上加入 front 高置信度 strand map；不积分发束",
        "inputs": {
            "front_image": str(args.front_image),
            "front_strand": str(args.front_strand),
            "front_seg": str(args.front_seg),
            "bank_selection": str(args.bank_selection),
        },
        "observation_policy": {
            "strand_encoding": "R=hair, G=(dy+1)/2, B=(-dx+1)/2",
            "confidence": "double-angle coherence * seg boundary distance * encoding magnitude * front-facing",
            "confidence_threshold": float(args.confidence_threshold),
            "hidden_confidence_forced_zero": True,
            "bank_mode": "hard Dirichlet beta=60deg",
            "root_generation_enabled": False,
            "future_root_seeding_requires_seg": True,
        },
        "observation_vertex_counts": observation_counts,
        "weight_search": candidate_reports,
        "selected": {
            "screening_length_mm": selected["screening_length_m"] * 1000.0,
            "metrics": selected_metrics,
            "charts": {
                chart_id: chart["metrics"] for chart_id, chart in selected["charts"].items()
            },
        },
        "summary": {
            "baseline_front_median_error_deg": selected_metrics["baseline_error_median_deg"],
            "observed_front_median_error_deg": selected_metrics["observed_error_median_deg"],
            "median_improvement_deg": selected_metrics["median_improvement_deg"],
            "observed_front_q95_error_deg": selected_metrics["observed_error_q95_deg"],
            "max_bank_dirichlet_residual": float(max_bank_residual),
            "hidden_region_max_confidence": float(hidden_max_confidence),
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "front_raw_preview": str(args.output_dir / "observed_field_front_raw_comparison.png"),
            "preview_3d": str(args.output_dir / "observed_field_3d_preview.png"),
            "data": str(args.output_dir / "observed_fields.npz"),
            "report": str(args.output_dir / "step_05_observed_field_report.json"),
        },
    }
    (args.output_dir / "step_05_observed_field_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_step5_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    step2_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_02_cut_atlas"
    step4_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_04_bank_only_field"
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--step2-report", type=Path, default=step2_dir / "step_02_cut_atlas_report.json")
    parser.add_argument("--step4-report", type=Path, default=step4_dir / "step_04_bank_only_field_report.json")
    parser.add_argument("--bank-selection", type=Path, default=step4_dir / "selected_bank_boundary.json")
    parser.add_argument("--bank-fields", type=Path, default=step4_dir / "bank_only_fields.npz")
    parser.add_argument("--atlas-data", type=Path, default=step2_dir / "cut_atlas_data.npz")
    parser.add_argument("--left-atlas", type=Path, default=step2_dir / "left_atlas.obj")
    parser.add_argument("--right-atlas", type=Path, default=step2_dir / "right_atlas.obj")
    parser.add_argument("--cap-mesh", type=Path, default=step2_dir / "cap.obj")
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-strand", type=Path, default=DEFAULT_ROOT / "maps/strand_map/front.png")
    parser.add_argument("--front-seg", type=Path, default=DEFAULT_ROOT / "maps/seg/front.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--screening-length-mm", type=float, nargs="+", default=[4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_05_observed_field",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_observed_field_step(parse_step5_args())
    print(
        json.dumps(
            {
                "passed_numeric_gates": result["passed_numeric_gates"],
                "human_review_status": result["human_review_status"],
                "report": result["outputs"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
