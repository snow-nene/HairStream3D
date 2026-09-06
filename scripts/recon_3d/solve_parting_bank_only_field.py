#!/usr/bin/env python3
"""Step 4：仅用发缝两岸边界求解无符号 connection Laplace–Beltrami 线场。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.recon_3d.validate_parting_connection_lb import (
    build_vertex_frames,
    cotangent_connection_operator,
    solve_dirichlet_system,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def calibration_from_param(path: Path) -> np.ndarray:
    """把现有正交相机参数转换为 4x4 投影矩阵。"""
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


def project_points_to_image(
    points: np.ndarray, calib: np.ndarray, image_shape: tuple[int, ...]
) -> np.ndarray:
    """把三维点投影为 raw_img 像素坐标。"""
    height, width = image_shape[:2]
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ calib.T
    ndc = clip[:, :2] / clip[:, 3:4]
    return np.column_stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * (width - 1),
            (ndc[:, 1] + 1.0) * 0.5 * (height - 1),
        ]
    )


def load_step4_inputs(args: argparse.Namespace) -> dict:
    """验证 Step 2/3 门禁并读取双 atlas、岸线、发缝曲线与 cap。"""
    step2_report = json.loads(args.step2_report.read_text(encoding="utf-8"))
    step3_report = json.loads(args.step3_report.read_text(encoding="utf-8"))
    if not step2_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 2 数值门禁未通过")
    if not step3_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 3 数值门禁未通过")
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
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取正面原图: {args.front_image}")
    calib = calibration_from_param(args.front_calib)
    depth_calib = calibration_from_param(args.front_depth_calib)
    calib[2] = depth_calib[2]
    return {
        "step2_report": step2_report,
        "step3_report": step3_report,
        "meshes": meshes,
        "cap": cap,
        "curve": curve,
        "shores": shores,
        "image": image,
        "calib": calib,
    }


def match_bank_vertices(mesh: trimesh.Trimesh, shore_points: np.ndarray) -> tuple[np.ndarray, dict]:
    """将 Step 2 连续裁剪生成的岸线点匹配到 atlas 边界顶点。"""
    distances, vertex_ids = cKDTree(np.asarray(mesh.vertices)).query(shore_points, k=1)
    unique_ids = np.unique(vertex_ids.astype(np.int64))
    if float(np.max(distances)) > 1e-6:
        raise RuntimeError(f"岸线到 atlas 的匹配误差过大: {float(np.max(distances)):.3e} m")
    return unique_ids, {
        "shore_point_count": int(len(shore_points)),
        "matched_vertex_count": int(len(unique_ids)),
        "max_distance_m": float(np.max(distances)),
        "q95_distance_m": float(np.quantile(distances, 0.95)),
    }


def build_bank_boundary_values(
    mesh: trimesh.Trimesh,
    frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    bank_vertex_ids: np.ndarray,
    curve: np.ndarray,
    beta_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """由曲线切向和朝本岸的曲面侧向构造物理边界切线及二重角值。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    curve_tree = cKDTree(curve)
    _, nearest = curve_tree.query(vertices[bank_vertex_ids], k=1)
    previous = np.maximum(nearest - 1, 0)
    following = np.minimum(nearest + 1, len(curve) - 1)
    normals = frames[2][bank_vertex_ids]
    tau = curve[following] - curve[previous]
    tau -= np.sum(tau * normals, axis=1, keepdims=True) * normals
    tau /= np.maximum(np.linalg.norm(tau, axis=1, keepdims=True), 1e-15)

    displacement = vertices[bank_vertex_ids] - curve[nearest]
    lateral = displacement - np.sum(displacement * normals, axis=1, keepdims=True) * normals
    lateral -= np.sum(lateral * tau, axis=1, keepdims=True) * tau
    lateral_norm = np.linalg.norm(lateral, axis=1)
    weak = lateral_norm < 1e-10
    lateral[weak] = np.cross(normals[weak], tau[weak])
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-15)
    alignment = np.sum(lateral * displacement, axis=1)
    lateral[alignment < 0.0] *= -1.0

    beta = np.deg2rad(float(beta_deg))
    tangents = np.cos(beta) * tau + np.sin(beta) * lateral
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-15)
    local = (
        np.sum(tangents * frames[0][bank_vertex_ids], axis=1)
        + 1j * np.sum(tangents * frames[1][bank_vertex_ids], axis=1)
    )
    return local * local, tangents


def recover_line_tangents(
    q: np.ndarray, frames: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """从二重角场恢复一个无符号三维切线代表。"""
    theta = 0.5 * np.angle(q)
    tangent = np.cos(theta)[:, None] * frames[0] + np.sin(theta)[:, None] * frames[1]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    return tangent


def line_field_singularity_candidates(
    mesh: trimesh.Trimesh,
    q: np.ndarray,
    operator: dict,
    magnitude_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    """把低模长连通区标为候选，并估计代表点的一环线场指数。"""
    magnitude = np.abs(q)
    low = magnitude < float(magnitude_threshold)
    edges = operator["edges"]
    adjacency = sparse.csr_matrix(
        (
            np.ones(2 * len(edges), dtype=np.int8),
            (
                np.concatenate([edges[:, 0], edges[:, 1]]),
                np.concatenate([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(len(magnitude), len(magnitude)),
    )
    candidate_mask = np.zeros(len(magnitude), dtype=bool)
    candidates: list[dict] = []
    low_ids = np.flatnonzero(low)
    if not len(low_ids):
        return candidate_mask, candidates
    low_graph = adjacency[low_ids][:, low_ids]
    component_count, labels = connected_components(low_graph, directed=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    e1, e2, _ = operator["frames"]
    transport_lookup = {}
    for edge, transport in zip(edges, operator["transports"]):
        i, j = map(int, edge)
        transport_lookup[(i, j)] = transport
        transport_lookup[(j, i)] = np.conj(transport)
    for component_id in range(component_count):
        ids = low_ids[labels == component_id]
        candidate_mask[ids] = True
        representative = int(ids[np.argmin(magnitude[ids])])
        neighbors = adjacency.indices[adjacency.indptr[representative] : adjacency.indptr[representative + 1]]
        offsets = vertices[neighbors] - vertices[representative]
        order = np.argsort(
            np.arctan2(offsets @ e2[representative], offsets @ e1[representative])
        )
        neighbors = neighbors[order]
        transported = np.asarray(
            [transport_lookup[(representative, int(neighbor))] * q[neighbor] for neighbor in neighbors]
        )
        if len(transported) >= 3 and np.all(np.abs(transported) > 1e-12):
            phases = np.angle(transported)
            wrapped_steps = np.angle(np.exp(1j * np.diff(np.r_[phases, phases[0]])))
            line_index = float(np.sum(wrapped_steps) / (4.0 * np.pi))
        else:
            line_index = 0.0
        candidates.append(
            {
                "representative_vertex": representative,
                "position": [float(value) for value in vertices[representative]],
                "component_vertex_count": int(len(ids)),
                "minimum_magnitude": float(magnitude[representative]),
                "line_index_estimate": line_index,
                "confidence": float(np.clip(1.0 - magnitude[representative] / magnitude_threshold, 0.0, 1.0)),
            }
        )
    return candidate_mask, candidates


def solve_bank_only_candidate(
    chart_id: str,
    mesh: trimesh.Trimesh,
    shore_points: np.ndarray,
    curve: np.ndarray,
    beta_deg: float,
    magnitude_threshold: float,
) -> dict:
    """求解单张 atlas、单个 beta 的岸线 Dirichlet connection 场。"""
    operator = cotangent_connection_operator(mesh)
    bank_ids, match_metrics = match_bank_vertices(mesh, shore_points)
    bank_q, bank_tangents = build_bank_boundary_values(
        mesh, operator["frames"], bank_ids, curve, beta_deg
    )
    q, solve_metrics = solve_dirichlet_system(operator["stiffness"], bank_ids, bank_q)
    flipped_bank_q, _ = build_bank_boundary_values(
        mesh, operator["frames"], bank_ids, curve, beta_deg
    )
    flipped_bank_q = (-np.sqrt(flipped_bank_q)) ** 2
    flipped_q, _ = solve_dirichlet_system(
        operator["stiffness"], bank_ids, flipped_bank_q
    )
    sign_flip_difference = float(np.max(np.abs(q - flipped_q), initial=0.0))
    candidate_mask, candidates = line_field_singularity_candidates(
        mesh, q, operator, magnitude_threshold
    )
    magnitude = np.abs(q)
    regular = ~candidate_mask
    regular_minimum = float(np.min(magnitude[regular])) if np.any(regular) else float("inf")
    normalized_q = q / np.maximum(magnitude, 1e-15)
    return {
        "chart_id": chart_id,
        "mesh": mesh,
        "operator": operator,
        "bank_ids": bank_ids,
        "bank_tangents": bank_tangents,
        "q": q,
        "normalized_q": normalized_q,
        "magnitude": magnitude,
        "candidate_mask": candidate_mask,
        "metrics": {
            "bank_match": match_metrics,
            "solve": solve_metrics,
            "sign_flip_max_q_difference": sign_flip_difference,
            "magnitude": {
                "minimum": float(np.min(magnitude)),
                "median": float(np.median(magnitude)),
                "q05": float(np.quantile(magnitude, 0.05)),
                "regular_region_minimum": regular_minimum,
                "threshold": float(magnitude_threshold),
            },
            "singularity_candidate_count": int(len(candidates)),
            "singularity_candidates": candidates,
        },
    }


def render_bank_candidate_preview(
    path: Path,
    candidates: dict[float, dict[str, dict]],
    cap: trimesh.Trimesh,
) -> None:
    """并排绘制三个 beta 的无向线场、岸线箭头和模长。"""
    beta_values = sorted(candidates)
    figure = plt.figure(figsize=(18, 11), dpi=180)
    colors = {"left": "#20a4f3", "right": "#ff9f1c"}
    cap_vertices = np.asarray(cap.vertices)
    cap_faces = np.asarray(cap.faces)
    all_vertices = np.concatenate(
        [np.asarray(result["mesh"].vertices) for result in candidates[beta_values[0]].values()]
    )
    center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
    radius = 0.55 * float(np.max(np.ptp(all_vertices, axis=0)))
    amplitude_scatter = None
    for column, beta in enumerate(beta_values):
        field_axis = figure.add_subplot(2, 3, column + 1, projection="3d")
        magnitude_axis = figure.add_subplot(2, 3, column + 4, projection="3d")
        field_axis.set_title(f"bank-only field, beta={beta:g} deg")
        magnitude_axis.set_title(f"|q| and candidates, beta={beta:g} deg")
        for axis in (field_axis, magnitude_axis):
            axis.set_axis_off()
            axis.view_init(elev=20, azim=-82)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.add_collection3d(
                Poly3DCollection(cap_vertices[cap_faces], facecolor="#ee3dad", alpha=0.55, edgecolor="none")
            )
        for chart_id, result in candidates[beta].items():
            mesh = result["mesh"]
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            field_axis.add_collection3d(
                Poly3DCollection(vertices[faces], facecolor=colors[chart_id], alpha=0.08, edgecolor="none")
            )
            sample = np.arange(0, len(vertices), max(1, len(vertices) // 210))
            tangents = recover_line_tangents(result["normalized_q"], result["operator"]["frames"])
            scale = 0.0032
            for first, second in zip(vertices[sample] - scale * tangents[sample], vertices[sample] + scale * tangents[sample]):
                field_axis.plot(*np.stack([first, second]).T, color=colors[chart_id], linewidth=0.65)
            bank_sample = result["bank_ids"][:: max(1, len(result["bank_ids"]) // 18)]
            bank_lookup = {int(vertex_id): index for index, vertex_id in enumerate(result["bank_ids"])}
            bank_tangents = np.asarray([result["bank_tangents"][bank_lookup[int(vertex_id)]] for vertex_id in bank_sample])
            field_axis.quiver(
                vertices[bank_sample, 0], vertices[bank_sample, 1], vertices[bank_sample, 2],
                bank_tangents[:, 0], bank_tangents[:, 1], bank_tangents[:, 2],
                length=0.008, normalize=True, color=colors[chart_id], linewidth=1.5,
            )
            amplitude_scatter = magnitude_axis.scatter(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], c=result["magnitude"],
                cmap="viridis", vmin=0.0, vmax=1.0, s=3,
            )
            singular_ids = np.flatnonzero(result["candidate_mask"])
            if len(singular_ids):
                magnitude_axis.scatter(
                    vertices[singular_ids, 0], vertices[singular_ids, 1], vertices[singular_ids, 2],
                    color="#ff1744", marker="x", s=18,
                )
    if amplitude_scatter is not None:
        figure.colorbar(amplitude_scatter, ax=figure.axes, shrink=0.45, pad=0.015, label="|q|")
    figure.suptitle("Step 4: bank-only unoriented connection fields (magenta = cap, red = low-|q| candidate)")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_front_bank_comparison(
    path: Path,
    candidates: dict[float, dict[str, dict]],
    cap: trimesh.Trimesh,
    image: np.ndarray,
    calib: np.ndarray,
) -> None:
    """把三个 beta 方向场投影叠加到 raw_img，而非 Blender 渲染图。"""
    panels = []
    colors = {"left": (255, 170, 20), "right": (20, 155, 255)}
    cap_pixels = project_points_to_image(np.asarray(cap.vertices), calib, image.shape)
    for beta in sorted(candidates):
        canvas = image.copy()
        cap_layer = canvas.copy()
        for face in np.asarray(cap.faces):
            cv2.fillConvexPoly(
                cap_layer,
                np.rint(cap_pixels[face]).astype(np.int32),
                (220, 50, 220),
                lineType=cv2.LINE_AA,
            )
        canvas = cv2.addWeighted(cap_layer, 0.45, canvas, 0.55, 0)
        for chart_id, result in candidates[beta].items():
            vertices = np.asarray(result["mesh"].vertices)
            tangents = recover_line_tangents(
                result["normalized_q"], result["operator"]["frames"]
            )
            sample = np.arange(0, len(vertices), max(1, len(vertices) // 230))
            scale = 0.0032
            first = project_points_to_image(
                vertices[sample] - scale * tangents[sample], calib, image.shape
            )
            second = project_points_to_image(
                vertices[sample] + scale * tangents[sample], calib, image.shape
            )
            for start, stop in zip(first, second):
                cv2.line(
                    canvas,
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(stop).astype(int)),
                    colors[chart_id],
                    1,
                    cv2.LINE_AA,
                )
            bank_sample = result["bank_ids"][:: max(1, len(result["bank_ids"]) // 20)]
            bank_lookup = {
                int(vertex_id): index for index, vertex_id in enumerate(result["bank_ids"])
            }
            bank_tangent = np.asarray(
                [result["bank_tangents"][bank_lookup[int(vertex_id)]] for vertex_id in bank_sample]
            )
            bank_start = project_points_to_image(vertices[bank_sample], calib, image.shape)
            bank_stop = project_points_to_image(
                vertices[bank_sample] + 0.008 * bank_tangent, calib, image.shape
            )
            for start, stop in zip(bank_start, bank_stop):
                cv2.arrowedLine(
                    canvas,
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(stop).astype(int)),
                    colors[chart_id],
                    2,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )
        label = f"raw_img.png  bank-only beta={beta:g} deg"
        cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(canvas)
    comparison = np.concatenate(panels, axis=1)
    if not cv2.imwrite(str(path), comparison):
        raise RuntimeError(f"无法写出正面比较图: {path}")


def run_bank_only_field_step(args: argparse.Namespace) -> dict:
    """运行三个 beta 候选，执行门禁并输出人工检查材料。"""
    inputs = load_step4_inputs(args)
    beta_values = [float(value) for value in args.beta]
    candidates: dict[float, dict[str, dict]] = {}
    for beta in beta_values:
        candidates[beta] = {
            chart_id: solve_bank_only_candidate(
                chart_id,
                mesh,
                inputs["shores"][chart_id],
                inputs["curve"],
                beta,
                args.magnitude_threshold,
            )
            for chart_id, mesh in inputs["meshes"].items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_bank_candidate_preview(
        args.output_dir / "bank_only_beta_comparison.png", candidates, inputs["cap"]
    )
    render_front_bank_comparison(
        args.output_dir / "bank_only_beta_front_raw_comparison.png",
        candidates,
        inputs["cap"],
        inputs["image"],
        inputs["calib"],
    )
    arrays = {}
    for beta, charts in candidates.items():
        beta_key = f"beta_{int(round(beta))}"
        for chart_id, result in charts.items():
            arrays[f"{beta_key}_{chart_id}_q"] = result["q"].astype(np.complex64)
            arrays[f"{beta_key}_{chart_id}_magnitude"] = result["magnitude"].astype(np.float32)
            arrays[f"{beta_key}_{chart_id}_candidate_mask"] = result["candidate_mask"]
    np.savez_compressed(args.output_dir / "bank_only_fields.npz", **arrays)

    candidate_metrics = {
        f"beta_{int(round(beta))}": {
            chart_id: result["metrics"] for chart_id, result in charts.items()
        }
        for beta, charts in candidates.items()
    }
    all_metrics = [
        metrics
        for beta_metrics in candidate_metrics.values()
        for metrics in beta_metrics.values()
    ]
    max_sign_flip = max(item["sign_flip_max_q_difference"] for item in all_metrics)
    max_dirichlet = max(item["solve"]["dirichlet_max_residual"] for item in all_metrics)
    regular_minimum = min(item["magnitude"]["regular_region_minimum"] for item in all_metrics)
    cross_atlas = int(inputs["step2_report"]["cross_atlas_adjacency_outside_cap"])
    numeric_gates = {
        "sign_flip_q_difference_le_1e-8": max_sign_flip <= 1e-8,
        "dirichlet_residual_le_1e-8": max_dirichlet <= 1e-8,
        "cross_atlas_coupling_outside_cap_eq_0": cross_atlas == 0,
        "regular_region_magnitude_ge_threshold": regular_minimum + 1e-12 >= float(args.magnitude_threshold),
    }
    report = {
        "image_id": args.image_id,
        "step": "step_04_bank_only_field",
        "purpose": "只用两岸物理边界验证无符号线场的全局平滑传播；不使用 strand map，不生成发丝",
        "parent_steps": {
            "step2_report": str(args.step2_report),
            "step3_report": str(args.step3_report),
            "continued_by_user": True,
        },
        "domain_policy": {
            "pde_domain_may_include_forehead_buffer": True,
            "root_generation_enabled": False,
            "future_root_seeding_requires_seg": True,
        },
        "config": {
            "beta_deg": beta_values,
            "magnitude_threshold": float(args.magnitude_threshold),
            "observation": "none",
            "boundary_condition": "bank Dirichlet; other boundaries natural Neumann",
        },
        "candidates": candidate_metrics,
        "summary": {
            "max_sign_flip_q_difference": float(max_sign_flip),
            "max_dirichlet_residual": float(max_dirichlet),
            "regular_region_minimum_magnitude": float(regular_minimum),
            "cross_atlas_coupling_outside_cap": cross_atlas,
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "preview": str(args.output_dir / "bank_only_beta_comparison.png"),
            "front_raw_preview": str(args.output_dir / "bank_only_beta_front_raw_comparison.png"),
            "data": str(args.output_dir / "bank_only_fields.npz"),
            "report": str(args.output_dir / "step_04_bank_only_field_report.json"),
        },
    }
    (args.output_dir / "step_04_bank_only_field_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_step4_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    step2_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_02_cut_atlas"
    step3_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_03_operator_validation"
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--step2-report", type=Path, default=step2_dir / "step_02_cut_atlas_report.json")
    parser.add_argument("--step3-report", type=Path, default=step3_dir / "step_03_operator_validation_report.json")
    parser.add_argument("--atlas-data", type=Path, default=step2_dir / "cut_atlas_data.npz")
    parser.add_argument("--left-atlas", type=Path, default=step2_dir / "left_atlas.obj")
    parser.add_argument("--right-atlas", type=Path, default=step2_dir / "right_atlas.obj")
    parser.add_argument("--cap-mesh", type=Path, default=step2_dir / "cap.obj")
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--beta", type=float, nargs="+", default=[30.0, 45.0, 60.0])
    parser.add_argument("--magnitude-threshold", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_04_bank_only_field",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_bank_only_field_step(parse_step4_args())
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
