#!/usr/bin/env python3
"""Step 3：验证双 atlas 上的 cotangent connection Laplace–Beltrami 算子。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial import cKDTree
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step3_meshes(args: argparse.Namespace) -> dict[str, trimesh.Trimesh]:
    """读取通过 Step 2 数值门禁的左右独立 atlas。"""
    report = json.loads(args.step2_report.read_text(encoding="utf-8"))
    if not report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 2 数值门禁未通过，禁止验证联络算子")
    meshes = {}
    for chart_id, path in (("left", args.left_atlas), ("right", args.right_atlas)):
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"{chart_id} atlas 不是有效三角网格")
        meshes[chart_id] = mesh
    return meshes


def build_vertex_frames(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为每个顶点构造右手正交切向标架。"""
    normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    axes = np.eye(3, dtype=np.float64)
    axis_ids = np.argmin(np.abs(normals @ axes.T), axis=1)
    references = axes[axis_ids]
    e1 = references - np.sum(references * normals, axis=1, keepdims=True) * normals
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-15)
    e2 = np.cross(normals, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-15)
    return e1, e2, normals


def minimal_rotation_transport(source_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    """返回把 source 法向最短旋转到 target 法向的三维旋转矩阵。"""
    cross = np.cross(source_normal, target_normal)
    cosine = float(np.clip(np.dot(source_normal, target_normal), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-14:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.eye(3)[int(np.argmin(np.abs(source_normal)))]
        axis -= np.dot(axis, source_normal) * source_normal
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


def cotangent_connection_operator(mesh: trimesh.Trimesh) -> dict:
    """组装 lumped mass、cotangent stiffness 与二重角联络矩阵。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    e1, e2, normals = build_vertex_frames(mesh)
    mass = np.zeros(len(vertices), dtype=np.float64)
    weights: dict[tuple[int, int], float] = {}
    for face in faces:
        triangle = vertices[face]
        twice_area = float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
        if twice_area <= 1e-16:
            continue
        mass[face] += twice_area / 6.0
        for local_a, local_b, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            a = int(face[local_a])
            b = int(face[local_b])
            edge = (min(a, b), max(a, b))
            va = triangle[local_a] - triangle[opposite]
            vb = triangle[local_b] - triangle[opposite]
            cotangent = float(np.dot(va, vb) / twice_area)
            weights[edge] = weights.get(edge, 0.0) + 0.5 * cotangent

    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    transports = []
    edge_array = []
    for (i, j), weight in weights.items():
        rotation = minimal_rotation_transport(normals[j], normals[i])
        transported_e1 = rotation @ e1[j]
        rho = np.arctan2(np.dot(transported_e1, e2[i]), np.dot(transported_e1, e1[i]))
        transport_ji = np.exp(2j * rho)
        rows.extend([i, j, i, j])
        cols.extend([i, j, j, i])
        values.extend([weight, weight, -weight * transport_ji, -weight * np.conj(transport_ji)])
        transports.append(transport_ji)
        edge_array.append((i, j))
    stiffness = sparse.csr_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    )
    return {
        "stiffness": stiffness,
        "mass": mass,
        "frames": (e1, e2, normals),
        "edges": np.asarray(edge_array, dtype=np.int64),
        "transports": np.asarray(transports, dtype=np.complex128),
        "cotangent_weights": np.asarray([weights[tuple(edge)] for edge in edge_array]),
    }


def graph_inverse_length_operator(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """构造旧式 1/edge_length、无切平面联络的图拉普拉斯。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    weights = 1.0 / np.maximum(
        np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1), 1e-12
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights, -weights, -weights])
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(vertices), len(vertices)))


def boundary_vertex_ids(mesh: trimesh.Trimesh) -> np.ndarray:
    """返回只属于一个三角面的拓扑边界顶点。"""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique_edges[counts == 1])


def solve_dirichlet_system(
    operator: sparse.csr_matrix, boundary_ids: np.ndarray, boundary_values: np.ndarray
) -> tuple[np.ndarray, dict]:
    """通过精确消元求解复数 Dirichlet 系统。"""
    vertex_count = operator.shape[0]
    fixed = np.zeros(vertex_count, dtype=bool)
    fixed[boundary_ids] = True
    free_ids = np.flatnonzero(~fixed)
    solution = np.zeros(vertex_count, dtype=np.complex128)
    solution[boundary_ids] = boundary_values
    if len(free_ids):
        lhs = operator[free_ids][:, free_ids]
        rhs = -(operator[free_ids][:, boundary_ids] @ boundary_values)
        solution[free_ids] = spsolve(lhs, rhs)
        equation_residual = np.linalg.norm(lhs @ solution[free_ids] - rhs) / max(
            np.linalg.norm(rhs), 1e-15
        )
    else:
        equation_residual = 0.0
    dirichlet_residual = float(
        np.max(np.abs(solution[boundary_ids] - boundary_values), initial=0.0)
    )
    return solution, {
        "free_vertex_count": int(len(free_ids)),
        "boundary_vertex_count": int(len(boundary_ids)),
        "equation_relative_residual": float(equation_residual),
        "dirichlet_max_residual": dirichlet_residual,
    }


def manufactured_line_field(
    frames: tuple[np.ndarray, np.ndarray, np.ndarray],
    guide: np.ndarray = np.array([0.15, -1.0, 0.2]),
) -> np.ndarray:
    """把固定三维方向投影到各切平面，构造无符号制造边界场。"""
    e1, e2, normals = frames
    guide = np.asarray(guide, dtype=np.float64)
    guide /= np.linalg.norm(guide)
    tangents = guide - np.sum(normals * guide, axis=1, keepdims=True) * normals
    tangent_norm = np.linalg.norm(tangents, axis=1)
    weak = tangent_norm < 1e-8
    if np.any(weak):
        fallback = np.array([1.0, 0.0, 0.0])
        tangents[weak] = fallback - np.sum(normals[weak] * fallback, axis=1, keepdims=True) * normals[weak]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-15)
    local = np.sum(tangents * e1, axis=1) + 1j * np.sum(tangents * e2, axis=1)
    return local * local


def angular_line_difference_deg(
    q_a: np.ndarray,
    frames_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    q_b: np.ndarray,
    frames_b: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """以三维无符号切线计算两个二重角场的夹角。"""
    def recover(q: np.ndarray, frames: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        theta = 0.5 * np.angle(q)
        return np.cos(theta)[:, None] * frames[0] + np.sin(theta)[:, None] * frames[1]

    tangent_a = recover(q_a, frames_a)
    tangent_b = recover(q_b, frames_b)
    cosine = np.clip(np.abs(np.sum(tangent_a * tangent_b, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def validate_transport() -> dict:
    """在平面制造网格上验证常值线场与联络传输。"""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    operator = cotangent_connection_operator(mesh)
    q = manufactured_line_field(operator["frames"], np.array([1.0, 0.25, 0.0]))
    residual = np.linalg.norm(operator["stiffness"] @ q) / max(
        np.linalg.norm(operator["stiffness"].data), 1e-15
    )
    unit_modulus_error = np.max(np.abs(np.abs(operator["transports"]) - 1.0), initial=0.0)
    return {
        "planar_constant_field_relative_residual": float(residual),
        "transport_unit_modulus_max_error": float(unit_modulus_error),
    }


def validate_atlas_operator(chart_id: str, mesh: trimesh.Trimesh) -> dict:
    """验证单张 atlas 的算子、制造解、旧图算子对照和细分收敛。"""
    operator = cotangent_connection_operator(mesh)
    stiffness = operator["stiffness"]
    hermitian_error = np.linalg.norm((stiffness - stiffness.getH()).data) / max(
        np.linalg.norm(stiffness.data), 1e-15
    )
    mass_matrix = sparse.diags(np.maximum(operator["mass"], 1e-15))
    eigen_count = min(6, max(1, stiffness.shape[0] - 2))
    eigenvalues = np.sort(
        np.real(eigsh(stiffness, k=eigen_count, M=mass_matrix, which="SM", return_eigenvectors=False))
    )
    boundary = boundary_vertex_ids(mesh)
    manufactured = manufactured_line_field(operator["frames"])
    connection_q, solve_metrics = solve_dirichlet_system(
        stiffness, boundary, manufactured[boundary]
    )
    graph_q, graph_solve_metrics = solve_dirichlet_system(
        graph_inverse_length_operator(mesh).astype(np.complex128),
        boundary,
        manufactured[boundary],
    )
    valid = (np.abs(connection_q) > 1e-10) & (np.abs(graph_q) > 1e-10)
    graph_difference = angular_line_difference_deg(
        connection_q[valid],
        tuple(frame[valid] for frame in operator["frames"]),
        graph_q[valid],
        tuple(frame[valid] for frame in operator["frames"]),
    )

    refined_vertices, refined_faces = trimesh.remesh.subdivide(
        np.asarray(mesh.vertices), np.asarray(mesh.faces)
    )
    refined_mesh = trimesh.Trimesh(vertices=refined_vertices, faces=refined_faces, process=False)
    refined_operator = cotangent_connection_operator(refined_mesh)
    refined_boundary = boundary_vertex_ids(refined_mesh)
    refined_manufactured = manufactured_line_field(refined_operator["frames"])
    refined_q, refined_solve_metrics = solve_dirichlet_system(
        refined_operator["stiffness"], refined_boundary, refined_manufactured[refined_boundary]
    )
    nearest_refined = cKDTree(np.asarray(refined_mesh.vertices)).query(
        np.asarray(mesh.vertices), k=1
    )[1]
    coarse_free = np.ones(len(mesh.vertices), dtype=bool)
    coarse_free[boundary] = False
    comparable = coarse_free & (np.abs(connection_q) > 1e-10) & (
        np.abs(refined_q[nearest_refined]) > 1e-10
    )
    refinement_difference = angular_line_difference_deg(
        connection_q[comparable],
        tuple(frame[comparable] for frame in operator["frames"]),
        refined_q[nearest_refined[comparable]],
        tuple(frame[nearest_refined[comparable]] for frame in refined_operator["frames"]),
    )
    cot_weights = operator["cotangent_weights"]
    return {
        "chart_id": chart_id,
        "mesh": mesh,
        "operator": operator,
        "connection_q": connection_q,
        "graph_q": graph_q,
        "graph_difference_deg": graph_difference,
        "refinement_difference_deg": refinement_difference,
        "metrics": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "boundary_vertices": int(len(boundary)),
            "hermitian_relative_error": float(hermitian_error),
            "smallest_generalized_eigenvalues": [float(value) for value in eigenvalues],
            "negative_cotangent_weight_count": int(np.count_nonzero(cot_weights < 0.0)),
            "cotangent_weight_min": float(np.min(cot_weights)),
            "connection_solve": solve_metrics,
            "graph_solve": graph_solve_metrics,
            "refined_solve": refined_solve_metrics,
            "graph_vs_connection_direction_deg": {
                "median": float(np.median(graph_difference)),
                "q95": float(np.quantile(graph_difference, 0.95)),
                "max": float(np.max(graph_difference)),
            },
            "refinement_direction_deg": {
                "median": float(np.median(refinement_difference)),
                "q95": float(np.quantile(refinement_difference, 0.95)),
                "max": float(np.max(refinement_difference)),
                "sample_count": int(len(refinement_difference)),
            },
        },
    }


def render_operator_preview(path: Path, results: dict[str, dict]) -> None:
    """绘制 connection 场、旧图场及二者方向差异。"""
    figure = plt.figure(figsize=(18, 6), dpi=180)
    titles = ["Connection cotangent field", "1/edge graph field", "Graph vs connection error"]
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    colors = {"left": "#2f8fce", "right": "#ff8b2b"}
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_axis_off()
        axis.view_init(elev=20, azim=-82)
    for chart_id, result in results.items():
        mesh = result["mesh"]
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        axes[0].add_collection3d(Poly3DCollection(vertices[faces], facecolor=colors[chart_id], alpha=0.12, edgecolor="none"))
        axes[1].add_collection3d(Poly3DCollection(vertices[faces], facecolor=colors[chart_id], alpha=0.12, edgecolor="none"))
        difference_full = angular_line_difference_deg(
            result["connection_q"], result["operator"]["frames"],
            result["graph_q"], result["operator"]["frames"],
        )
        scatter = axes[2].scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=difference_full, s=3, cmap="magma", vmin=0.0, vmax=max(15.0, float(np.quantile(difference_full, 0.95))))
        sample = np.arange(0, len(vertices), max(1, len(vertices) // 180))
        for axis, field in ((axes[0], result["connection_q"]), (axes[1], result["graph_q"])):
            theta = 0.5 * np.angle(field[sample])
            frames = result["operator"]["frames"]
            tangent = np.cos(theta)[:, None] * frames[0][sample] + np.sin(theta)[:, None] * frames[1][sample]
            scale = 0.0035
            start = vertices[sample] - scale * tangent
            stop = vertices[sample] + scale * tangent
            for first, second in zip(start, stop):
                axis.plot(*np.stack([first, second]).T, color=colors[chart_id], linewidth=0.65)
    all_vertices = np.concatenate([np.asarray(item["mesh"].vertices) for item in results.values()])
    center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
    radius = 0.55 * float(np.max(np.ptp(all_vertices, axis=0)))
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
    figure.colorbar(scatter, ax=axes[2], shrink=0.62, pad=0.02, label="line angle difference (deg)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_operator_validation_step(args: argparse.Namespace) -> dict:
    """执行 Step 3 数值验证并写出报告、数组和人工检查预览。"""
    meshes = load_step3_meshes(args)
    results = {
        chart_id: validate_atlas_operator(chart_id, mesh)
        for chart_id, mesh in meshes.items()
    }
    transport_metrics = validate_transport()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_operator_preview(args.output_dir / "connection_operator_preview.png", results)
    np.savez_compressed(
        args.output_dir / "operator_validation_data.npz",
        left_q=results["left"]["connection_q"].astype(np.complex64),
        right_q=results["right"]["connection_q"].astype(np.complex64),
        left_graph_q=results["left"]["graph_q"].astype(np.complex64),
        right_graph_q=results["right"]["graph_q"].astype(np.complex64),
        left_refinement_difference_deg=results["left"]["refinement_difference_deg"].astype(np.float32),
        right_refinement_difference_deg=results["right"]["refinement_difference_deg"].astype(np.float32),
    )
    chart_metrics = {chart_id: result["metrics"] for chart_id, result in results.items()}
    max_hermitian = max(item["hermitian_relative_error"] for item in chart_metrics.values())
    max_dirichlet = max(item["connection_solve"]["dirichlet_max_residual"] for item in chart_metrics.values())
    max_refinement_q95 = max(item["refinement_direction_deg"]["q95"] for item in chart_metrics.values())
    numeric_gates = {
        "hermitian_relative_error_le_1e-10": max_hermitian <= 1e-10,
        "dirichlet_max_residual_le_1e-8": max_dirichlet <= 1e-8,
        "refinement_direction_q95_le_5_deg": max_refinement_q95 <= 5.0,
        "planar_constant_field_residual_le_1e-10": transport_metrics["planar_constant_field_relative_residual"] <= 1e-10,
    }
    report = {
        "image_id": args.image_id,
        "step": "step_03_operator_validation",
        "purpose": "只验证 cotangent 二重角 connection Laplace–Beltrami 算子；不使用 strand map，不生成发丝",
        "parent_step": {
            "report": str(args.step2_report),
            "human_approval": "continued_by_user",
        },
        "domain_policy": {
            "pde_domain": "Step 2 crown atlas，可包含额头数值缓冲区",
            "growth_domain": "后续发根播种必须经过 seg/hair_growth_mask，本步骤不播种",
        },
        "operator": {
            "stiffness": "cotangent finite-element stiffness",
            "mass": "lumped barycentric vertex area",
            "field": "unoriented double-angle q=exp(2i theta)",
            "connection": "minimal normal-alignment rotation between adjacent tangent frames",
        },
        "transport_manufactured_test": transport_metrics,
        "charts": chart_metrics,
        "summary": {
            "max_hermitian_relative_error": float(max_hermitian),
            "max_dirichlet_residual": float(max_dirichlet),
            "max_refinement_q95_deg": float(max_refinement_q95),
        },
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "preview": str(args.output_dir / "connection_operator_preview.png"),
            "data": str(args.output_dir / "operator_validation_data.npz"),
            "report": str(args.output_dir / "step_03_operator_validation_report.json"),
        },
    }
    (args.output_dir / "step_03_operator_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_step3_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    step2_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_02_cut_atlas"
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--step2-report", type=Path, default=step2_dir / "step_02_cut_atlas_report.json")
    parser.add_argument("--left-atlas", type=Path, default=step2_dir / "left_atlas.obj")
    parser.add_argument("--right-atlas", type=Path, default=step2_dir / "right_atlas.obj")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_03_operator_validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_operator_validation_step(parse_step3_args())
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
