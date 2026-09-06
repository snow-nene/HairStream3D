#!/usr/bin/env python3
"""Step 2：从已确认的可见发缝曲线构造有限宽度空带与双 atlas。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.recon_3d.run_parting_topology_smoke import (
    crop_mesh_around_curve,
    curve_lateral_frame,
    load_curve,
)


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step2_inputs(args: argparse.Namespace) -> dict:
    """验证 Step 1 结果，并读取头模、原图标定、mask 和候选曲线。"""
    report = json.loads(args.step1_report.read_text(encoding="utf-8"))
    if not report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 1 数值门禁未通过，禁止构造 cut atlas")
    if report.get("recommended_curve") != "raw_mask_raycast":
        raise RuntimeError("Step 1 推荐曲线不是 raw_mask_raycast")

    with np.load(args.parting_curve, allow_pickle=False) as archive:
        curve = np.asarray(archive["points"], dtype=np.float64).reshape(-1, 3)
        source_name = str(archive["source_name"])
        review_status = str(archive["human_review_status"])
    if source_name != "raw_mask_raycast" or len(curve) < 3:
        raise RuntimeError("Step 1 候选来源或点数无效")

    mesh = trimesh.load(str(args.head_mesh), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("head_mesh 必须是非空三角网格")

    def calibration_from_param(path: Path) -> np.ndarray:
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

    calib = calibration_from_param(args.front_calib)
    depth_calib = calibration_from_param(args.front_depth_calib)
    calib[2] = depth_calib[2]
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(args.parting_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError("无法读取原图或发缝 mask")
    if mask.shape != image.shape[:2]:
        raise ValueError("原图与发缝 mask 尺寸不一致")

    return {
        "step1_report": report,
        "curve": curve,
        "curve_source": source_name,
        "human_review_status_before_step2": review_status,
        "mesh": mesh,
        "calib": calib,
        "image": image,
        "mask": mask > 127,
    }


def build_weighted_mesh_adjacency(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """构造按三维边长加权的无向曲面邻接。"""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([lengths, lengths])
    adjacency = sparse.csr_matrix(
        (weights, (rows, cols)), shape=(len(vertices), len(vertices))
    )
    return adjacency, edges


def compute_geodesic_parting_coordinates(
    scalp: trimesh.Trimesh, curve: np.ndarray, lateral: np.ndarray
) -> dict:
    """计算曲面测地距、左右符号、曲线弧长和后冠 cap 坐标。"""
    vertices = np.asarray(scalp.vertices, dtype=np.float64)
    faces = np.asarray(scalp.faces, dtype=np.int64)
    adjacency, edges = build_weighted_mesh_adjacency(vertices, faces)
    curve_tree = cKDTree(curve)
    _, nearest_curve = curve_tree.query(vertices)
    offset = vertices - curve[nearest_curve]
    lateral_sign_value = np.sum(offset * lateral[nearest_curve], axis=1)
    side = np.where(lateral_sign_value < 0.0, -1.0, 1.0)

    vertex_tree = cKDTree(vertices)
    seed_distance, seed_ids = vertex_tree.query(curve)
    unique_seeds = {}
    for vertex_id, distance_value in zip(seed_ids, seed_distance):
        vertex_id = int(vertex_id)
        unique_seeds[vertex_id] = min(
            unique_seeds.get(vertex_id, float("inf")), float(distance_value)
        )
    virtual = len(vertices)
    coo = adjacency.tocoo()
    seed_vertices = np.asarray(sorted(unique_seeds), dtype=np.int64)
    seed_weights = np.asarray(
        [unique_seeds[int(vertex_id)] for vertex_id in seed_vertices],
        dtype=np.float64,
    )
    rows = np.concatenate([coo.row, seed_vertices, np.full(len(seed_vertices), virtual)])
    cols = np.concatenate([coo.col, np.full(len(seed_vertices), virtual), seed_vertices])
    data = np.concatenate([coo.data, seed_weights, seed_weights])
    augmented = sparse.csr_matrix(
        (data, (rows, cols)), shape=(virtual + 1, virtual + 1)
    )
    geodesic = np.asarray(
        dijkstra(augmented, directed=False, indices=virtual), dtype=np.float64
    )[:virtual]

    curve_steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    curve_arclength = np.concatenate([[0.0], np.cumsum(curve_steps)])
    arclength = curve_arclength[nearest_curve]
    endpoint_distance = np.linalg.norm(vertices - curve[0], axis=1)
    tangent_sample = min(4, len(curve) - 1)
    start_tangent = curve[tangent_sample] - curve[0]
    start_tangent /= max(float(np.linalg.norm(start_tangent)), 1e-12)
    longitudinal = (vertices - curve[0]) @ start_tangent
    signed_geodesic = side * geodesic
    return {
        "edges": edges,
        "geodesic": geodesic,
        "signed_geodesic": signed_geodesic,
        "side": side.astype(np.int8),
        "nearest_curve": nearest_curve,
        "arclength": arclength,
        "endpoint_distance": endpoint_distance,
        "longitudinal": longitudinal,
        "seed_vertex_ids": seed_vertices,
        "seed_curve_distance": seed_weights,
    }


def clip_triangle_region(
    triangle: np.ndarray, constraint_values: np.ndarray
) -> list[np.ndarray]:
    """以线性标量约束 `value <= 0` 连续裁剪一个三角形。"""
    polygon = [
        (triangle[index].copy(), constraint_values[:, index].copy())
        for index in range(3)
    ]
    for constraint_id in range(constraint_values.shape[0]):
        if not polygon:
            break
        clipped = []
        for index, current in enumerate(polygon):
            previous = polygon[index - 1]
            current_inside = current[1][constraint_id] <= 1e-12
            previous_inside = previous[1][constraint_id] <= 1e-12
            if current_inside != previous_inside:
                previous_value = previous[1][constraint_id]
                current_value = current[1][constraint_id]
                weight = previous_value / (previous_value - current_value + 1e-30)
                position = previous[0] + weight * (current[0] - previous[0])
                attributes = previous[1] + weight * (current[1] - previous[1])
                clipped.append((position, attributes))
            if current_inside:
                clipped.append(current)
        polygon = clipped
    return [item[0] for item in polygon]


def build_clipped_region_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    constraints: list[np.ndarray],
    weld_tolerance: float = 1e-8,
) -> trimesh.Trimesh:
    """按若干顶点标量约束切分三角形，并在区域内焊接交点。"""
    output_vertices = []
    output_faces = []
    vertex_map = {}

    def add_vertex(point: np.ndarray) -> int:
        key = tuple(np.rint(point / weld_tolerance).astype(np.int64))
        if key not in vertex_map:
            vertex_map[key] = len(output_vertices)
            output_vertices.append(np.asarray(point, dtype=np.float64))
        return vertex_map[key]

    fields = np.asarray(constraints, dtype=np.float64)
    for face in faces:
        polygon = clip_triangle_region(vertices[face], fields[:, face])
        if len(polygon) < 3:
            continue
        ids = [add_vertex(point) for point in polygon]
        for index in range(1, len(ids) - 1):
            triangle = [ids[0], ids[index], ids[index + 1]]
            if len(set(triangle)) == 3:
                output_faces.append(triangle)
    if not output_faces:
        raise RuntimeError("裁剪区域为空")
    return trimesh.Trimesh(
        vertices=np.asarray(output_vertices),
        faces=np.asarray(output_faces, dtype=np.int64),
        process=False,
    )


def extract_iso_shore(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar: np.ndarray,
    endpoint_distance: np.ndarray,
    longitudinal: np.ndarray,
    cap_length: float,
) -> dict:
    """从原始三角形提取 cap 外零等值岸线，并验证其连续性。"""
    points = []
    segments = []
    point_map = {}
    tolerance = 1e-7

    def add_point(point: np.ndarray) -> int:
        key = tuple(np.rint(point / tolerance).astype(np.int64))
        if key not in point_map:
            point_map[key] = len(points)
            points.append(np.asarray(point, dtype=np.float64))
        return point_map[key]

    for face in faces:
        face_points = vertices[face]
        values = scalar[face]
        cap_values = endpoint_distance[face]
        longitudinal_values = longitudinal[face]
        intersections = []
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            first, second = values[start], values[stop]
            if first * second > 0.0:
                continue
            if abs(first - second) < 1e-15:
                continue
            weight = first / (first - second)
            if weight < -1e-9 or weight > 1.0 + 1e-9:
                continue
            cap_distance = cap_values[start] + weight * (
                cap_values[stop] - cap_values[start]
            )
            if cap_distance < float(cap_length):
                continue
            longitudinal_value = longitudinal_values[start] + weight * (
                longitudinal_values[stop] - longitudinal_values[start]
            )
            if longitudinal_value < 0.0:
                continue
            point = face_points[start] + weight * (
                face_points[stop] - face_points[start]
            )
            if not any(np.linalg.norm(point - old) < tolerance for old in intersections):
                intersections.append(point)
        if len(intersections) >= 2:
            if len(intersections) > 2:
                pair = max(
                    ((a, b) for a in range(len(intersections)) for b in range(a + 1, len(intersections))),
                    key=lambda pair_ids: np.linalg.norm(
                        intersections[pair_ids[0]] - intersections[pair_ids[1]]
                    ),
                )
                intersections = [intersections[pair[0]], intersections[pair[1]]]
            start_id, stop_id = map(add_point, intersections)
            if start_id != stop_id:
                segments.append([start_id, stop_id])

    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    segments_array = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    if len(segments_array) == 0:
        raise RuntimeError("岸线等值线为空")
    rows = np.concatenate([segments_array[:, 0], segments_array[:, 1]])
    cols = np.concatenate([segments_array[:, 1], segments_array[:, 0]])
    graph = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(points_array), len(points_array))
    )
    component_count, labels = connected_components(graph, directed=False)
    degree = np.asarray(graph.astype(bool).sum(axis=1)).reshape(-1)
    return {
        "points": points_array,
        "segments": segments_array,
        "component_count": int(component_count),
        "component_sizes": [int(np.sum(labels == label)) for label in range(component_count)],
        "endpoint_count": int(np.sum(degree == 1)),
        "branch_vertex_count": int(np.sum(degree > 2)),
        "continuous": bool(component_count == 1 and np.sum(degree > 2) == 0),
    }


def mesh_topology_metrics(mesh: trimesh.Trimesh) -> dict:
    """统计退化面、非流形边、边界与连通分量。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    repeated = np.any(
        np.column_stack(
            [faces[:, 0] == faces[:, 1], faces[:, 1] == faces[:, 2], faces[:, 2] == faces[:, 0]]
        ),
        axis=1,
    )
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    )
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    adjacency, _ = build_weighted_mesh_adjacency(vertices, faces)
    component_count, _ = connected_components(adjacency, directed=False)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "degenerate_faces": int(np.sum(repeated | (area <= 1e-12))),
        "nonmanifold_edges": int(np.sum(edge_counts > 2)),
        "boundary_edges": int(np.sum(edge_counts == 1)),
        "connected_components": int(component_count),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "area_m2": float(np.sum(area)),
    }


def write_obj_mesh(path: Path, mesh: trimesh.Trimesh) -> None:
    """写出不带材质的可检查 OBJ。"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    lines = [f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}" for point in vertices]
    lines.extend(
        f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}" for face in faces
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_front_atlas_preview(
    path: Path,
    image: np.ndarray,
    calib: np.ndarray,
    regions: dict,
    shores: dict,
) -> None:
    """把左右 atlas、空带、cap 和两条岸线叠加到 raw_img。"""
    canvas = image.copy()
    colors = {
        "left_atlas": (220, 120, 30),
        "right_atlas": (30, 170, 240),
        "gap_band": (35, 35, 35),
        "cap": (210, 40, 210),
    }
    alpha = {"left_atlas": 0.30, "right_atlas": 0.30, "gap_band": 0.55, "cap": 0.70}
    height, width = canvas.shape[:2]

    def project(points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack([points, np.ones(len(points))])
        clip = homogeneous @ calib.T
        ndc = clip[:, :2] / clip[:, 3:4]
        return np.column_stack(
            [(ndc[:, 0] + 1.0) * 0.5 * (width - 1), (ndc[:, 1] + 1.0) * 0.5 * (height - 1)]
        )

    for name in ("left_atlas", "right_atlas", "gap_band", "cap"):
        mesh = regions[name]
        pixels = project(np.asarray(mesh.vertices))
        layer = canvas.copy()
        for face in np.asarray(mesh.faces):
            polygon = np.rint(pixels[face]).astype(np.int32)
            cv2.fillConvexPoly(layer, polygon, colors[name], lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(layer, alpha[name], canvas, 1.0 - alpha[name], 0)

    for name, color in (("left", (255, 255, 0)), ("right", (0, 255, 255))):
        shore = shores[name]
        pixels = project(shore["points"])
        for edge in shore["segments"]:
            cv2.line(
                canvas,
                tuple(np.rint(pixels[edge[0]]).astype(int)),
                tuple(np.rint(pixels[edge[1]]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    legend = "blue=left atlas  orange=right atlas  black=4mm gap  magenta=cap"
    cv2.putText(canvas, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"无法写入预览: {path}")


def render_3d_atlas_preview(path: Path, regions: dict, shores: dict) -> None:
    """输出可旋转关系明确的静态三维 atlas 预览。"""
    figure = plt.figure(figsize=(10, 9))
    axis = figure.add_subplot(111, projection="3d")
    colors = {
        "left_atlas": "#1f77b4",
        "right_atlas": "#ff7f0e",
        "gap_band": "#303030",
        "cap": "#d627b5",
    }
    all_points = []
    for name, mesh in regions.items():
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        all_points.append(vertices)
        collection = Poly3DCollection(
            vertices[faces], facecolor=colors[name], edgecolor="none", alpha=0.82
        )
        axis.add_collection3d(collection)
    for name, color in (("left", "cyan"), ("right", "yellow")):
        shore = shores[name]
        for edge in shore["segments"]:
            segment = shore["points"][edge]
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=2.0)
    points = np.concatenate(all_points, axis=0)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.52 * float(np.max(points.max(axis=0) - points.min(axis=0)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-65)
    axis.set_title("Step 2: finite-width parting band and cut atlases")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_parting_cut_atlas_step(args: argparse.Namespace) -> dict:
    """构造并验证 Step 2 曲面空带、双 atlas、岸线和 cap。"""
    inputs = load_step2_inputs(args)
    curve, lateral = curve_lateral_frame(inputs["mesh"], inputs["curve"])
    scalp = crop_mesh_around_curve(
        inputs["mesh"], curve, args.crop_radius, args.vertical_margin
    )
    for _ in range(args.extra_subdivisions):
        vertices, faces = trimesh.remesh.subdivide(
            np.asarray(scalp.vertices), np.asarray(scalp.faces)
        )
        scalp = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    curve, lateral = curve_lateral_frame(scalp, curve)
    coordinates = compute_geodesic_parting_coordinates(scalp, curve, lateral)
    vertices = np.asarray(scalp.vertices, dtype=np.float64)
    faces = np.asarray(scalp.faces, dtype=np.int64)
    signed = coordinates["signed_geodesic"]
    endpoint = coordinates["endpoint_distance"]
    longitudinal = coordinates["longitudinal"]
    width = float(args.bank_width)
    outside_cap = float(args.cap_length) - endpoint

    left_phi = signed + width
    right_phi = width - signed
    band_left_phi = -signed - width
    band_right_phi = signed - width
    forward_phi = -longitudinal
    regions = {
        "left_atlas": build_clipped_region_mesh(vertices, faces, [left_phi, outside_cap]),
        "right_atlas": build_clipped_region_mesh(vertices, faces, [right_phi, outside_cap]),
        "gap_band": build_clipped_region_mesh(
            vertices,
            faces,
            [band_left_phi, band_right_phi, outside_cap, forward_phi],
        ),
        "cap": build_clipped_region_mesh(vertices, faces, [endpoint - float(args.cap_length)]),
    }
    shores = {
        "left": extract_iso_shore(
            vertices, faces, left_phi, endpoint, longitudinal, args.cap_length
        ),
        "right": extract_iso_shore(
            vertices, faces, right_phi, endpoint, longitudinal, args.cap_length
        ),
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_obj_mesh(output / "welded_crown_mesh.obj", scalp)
    for name, mesh in regions.items():
        write_obj_mesh(output / f"{name}.obj", mesh)
    np.savez_compressed(
        output / "cut_atlas_data.npz",
        curve=curve.astype(np.float32),
        scalp_vertices=vertices.astype(np.float32),
        scalp_faces=faces.astype(np.int32),
        signed_geodesic=signed.astype(np.float32),
        geodesic_distance=coordinates["geodesic"].astype(np.float32),
        endpoint_distance=endpoint.astype(np.float32),
        longitudinal=longitudinal.astype(np.float32),
        side=coordinates["side"],
        left_shore_points=shores["left"]["points"].astype(np.float32),
        left_shore_segments=shores["left"]["segments"].astype(np.int32),
        right_shore_points=shores["right"]["points"].astype(np.float32),
        right_shore_segments=shores["right"]["segments"].astype(np.int32),
        bank_width_m=np.asarray(width),
        cap_length_m=np.asarray(float(args.cap_length)),
    )
    render_front_atlas_preview(
        output / "cut_atlas_front_preview.png",
        inputs["image"],
        inputs["calib"],
        regions,
        shores,
    )
    render_3d_atlas_preview(
        output / "cut_atlas_3d_preview.png", regions, shores
    )

    topology = {"welded_crown": mesh_topology_metrics(scalp)}
    topology.update({name: mesh_topology_metrics(mesh) for name, mesh in regions.items()})
    edge_vertices = coordinates["edges"]
    edge_a = edge_vertices[:, 0]
    edge_b = edge_vertices[:, 1]
    outside_cap_edges = (endpoint[edge_a] >= float(args.cap_length)) & (
        endpoint[edge_b] >= float(args.cap_length)
    )
    left_vertices = signed <= -width
    right_vertices = signed >= width
    cross_atlas_edges = outside_cap_edges & (
        (left_vertices[edge_a] & right_vertices[edge_b])
        | (right_vertices[edge_a] & left_vertices[edge_b])
    )
    source_edges_spanning_both_banks = int(np.count_nonzero(cross_atlas_edges))
    quantization = 1e-8
    left_vertex_keys = {
        tuple(row)
        for row in np.rint(
            np.asarray(regions["left_atlas"].vertices) / quantization
        ).astype(np.int64)
    }
    right_vertex_keys = {
        tuple(row)
        for row in np.rint(
            np.asarray(regions["right_atlas"].vertices) / quantization
        ).astype(np.int64)
    }
    cross_atlas_adjacency_outside_cap = len(left_vertex_keys & right_vertex_keys)
    degenerate_total = int(sum(item["degenerate_faces"] for item in topology.values()))
    nonmanifold_total = int(sum(item["nonmanifold_edges"] for item in topology.values()))
    numeric_gates = {
        "cross_atlas_adjacency_outside_cap_eq_0": cross_atlas_adjacency_outside_cap == 0,
        "nonmanifold_edges_eq_0": nonmanifold_total == 0,
        "degenerate_faces_eq_0": degenerate_total == 0,
        "left_shore_continuous": shores["left"]["continuous"],
        "right_shore_continuous": shores["right"]["continuous"],
        "gap_band_connected": topology["gap_band"]["connected_components"] == 1,
    }
    report = {
        "image_id": args.image_id,
        "step": "step_02_cut_atlas",
        "purpose": "构造 4 mm 有限宽度发缝空带、左右独立 atlas 和后冠 cap；不求解方向场",
        "parent_step": {
            "report": str(args.step1_report),
            "candidate": str(args.parting_curve),
            "human_approval": "confirmed_by_user",
        },
        "inputs": {
            "head_mesh": str(args.head_mesh),
            "front_image": str(args.front_image),
            "front_calib": str(args.front_calib),
            "parting_mask": str(args.parting_mask),
        },
        "config": {
            "bank_width_m": width,
            "total_gap_width_m": 2.0 * width,
            "cap_length_m": float(args.cap_length),
            "crop_radius_m": float(args.crop_radius),
            "vertical_margin_m": float(args.vertical_margin),
            "extra_subdivisions": int(args.extra_subdivisions),
            "distance": "multi_source_mesh_edge_dijkstra_with_curve_seed_offsets",
            "triangle_split": "linear_scalar_clipping_at_geodesic_4mm_isolines",
        },
        "weld": {
            "source_vertices": int(len(inputs["mesh"].vertices)),
            "source_faces": int(len(inputs["mesh"].faces)),
            "crown_vertices": int(len(vertices)),
            "crown_faces": int(len(faces)),
            "winding_consistent": bool(scalp.is_winding_consistent),
        },
        "shore": {
            "left": {key: value for key, value in shores["left"].items() if key not in {"points", "segments"}},
            "right": {key: value for key, value in shores["right"].items() if key not in {"points", "segments"}},
        },
        "topology": topology,
        "cross_atlas_adjacency_outside_cap": cross_atlas_adjacency_outside_cap,
        "source_edges_spanning_both_bank_thresholds": source_edges_spanning_both_banks,
        "nonmanifold_edges_total": nonmanifold_total,
        "degenerate_faces_total": degenerate_total,
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "front_preview": str(output / "cut_atlas_front_preview.png"),
            "preview_3d": str(output / "cut_atlas_3d_preview.png"),
            "atlas_data": str(output / "cut_atlas_data.npz"),
            "left_atlas": str(output / "left_atlas.obj"),
            "right_atlas": str(output / "right_atlas.obj"),
            "gap_band": str(output / "gap_band.obj"),
            "cap": str(output / "cap.obj"),
        },
    }
    (output / "step_02_cut_atlas_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_step2_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--step1-report", type=Path, default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_01_visible_curve/curve_visibility_report.json")
    parser.add_argument("--parting-curve", type=Path, default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_01_visible_curve/accepted_curve_candidate.npz")
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--parting-mask", type=Path, default=DEFAULT_ROOT / "pde_governance/front_parting_dinov3_temp_rerun/parting_region_mask.png")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_02_cut_atlas")
    parser.add_argument("--bank-width", type=float, default=0.004)
    parser.add_argument("--cap-length", type=float, default=0.012)
    parser.add_argument("--crop-radius", type=float, default=0.08)
    parser.add_argument("--vertical-margin", type=float, default=0.04)
    parser.add_argument("--extra-subdivisions", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_parting_cut_atlas_step(parse_step2_args())
    print(
        json.dumps(
            {
                "passed_numeric_gates": result["passed_numeric_gates"],
                "human_review_status": result["human_review_status"],
                "report": result["outputs"]["atlas_data"].replace(
                    "cut_atlas_data.npz", "step_02_cut_atlas_report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
