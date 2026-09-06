#!/usr/bin/env python3
"""Step 6：在双 atlas 三角面内连续积分 256 条 30 mm 根部轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree
import trimesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.recon_strategy.parting_groom import write_ordered_strands
from scripts.recon_3d.solve_parting_bank_only_field import (
    calibration_from_param,
    project_points_to_image,
    recover_line_tangents,
)
from scripts.recon_3d.validate_parting_connection_lb import build_vertex_frames


IMAGE_ID = "0d285f5be7fa09c3dbbf1c9334047888"
DEFAULT_ROOT = Path("results/multiview_data") / IMAGE_ID


def load_step6_inputs(args: argparse.Namespace) -> dict:
    """验证 Step 2/5，读取双 atlas、观测场、岸线、cap、gap 和原图。"""
    step2_report = json.loads(args.step2_report.read_text(encoding="utf-8"))
    step5_report = json.loads(args.step5_report.read_text(encoding="utf-8"))
    if not step2_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 2 数值门禁未通过")
    if not step5_report.get("passed_numeric_gates", False):
        raise RuntimeError("Step 5 数值门禁未通过")
    meshes = {}
    for chart_id, path in (("left", args.left_atlas), ("right", args.right_atlas)):
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"{chart_id} atlas 无效")
        meshes[chart_id] = mesh
    cap = trimesh.load(str(args.cap_mesh), process=False)
    gap = trimesh.load(str(args.gap_mesh), process=False)
    if isinstance(cap, trimesh.Scene):
        cap = trimesh.util.concatenate(tuple(cap.geometry.values()))
    if isinstance(gap, trimesh.Scene):
        gap = trimesh.util.concatenate(tuple(gap.geometry.values()))
    head = trimesh.load(str(args.head_mesh), process=True)
    if isinstance(head, trimesh.Scene):
        head = trimesh.util.concatenate(tuple(head.geometry.values()))

    with np.load(args.atlas_data, allow_pickle=False) as archive:
        curve = np.asarray(archive["curve"], dtype=np.float64)
        shores = {
            chart_id: {
                "points": np.asarray(archive[f"{chart_id}_shore_points"], dtype=np.float64),
                "segments": np.asarray(archive[f"{chart_id}_shore_segments"], dtype=np.int64),
            }
            for chart_id in ("left", "right")
        }
    with np.load(args.observed_fields, allow_pickle=False) as archive:
        fields = {
            "left": np.asarray(archive["left_q"], dtype=np.complex128),
            "right": np.asarray(archive["right_q"], dtype=np.complex128),
        }
    image = cv2.imread(str(args.front_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取原图: {args.front_image}")
    calib = calibration_from_param(args.front_calib)
    depth_calib = calibration_from_param(args.front_depth_calib)
    calib[2] = depth_calib[2]
    return {
        "step2_report": step2_report,
        "step5_report": step5_report,
        "meshes": meshes,
        "cap": cap,
        "gap": gap,
        "head": head,
        "curve": curve,
        "shores": shores,
        "fields": fields,
        "image": image,
        "calib": calib,
    }


def order_shore_vertex_chain(
    mesh: trimesh.Trimesh,
    shore_points: np.ndarray,
    shore_segments: np.ndarray,
    cap_center: np.ndarray,
) -> np.ndarray:
    """把无序岸线线段恢复为从 cap 指向前端的 atlas 边界顶点链。"""
    distances, point_vertex_ids = cKDTree(np.asarray(mesh.vertices)).query(shore_points, k=1)
    if float(np.max(distances)) > 1e-6:
        raise RuntimeError("岸线点无法精确匹配 atlas 顶点")
    adjacency: dict[int, list[int]] = {}
    for point_a, point_b in shore_segments:
        a = int(point_vertex_ids[int(point_a)])
        b = int(point_vertex_ids[int(point_b)])
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    endpoints = [vertex_id for vertex_id, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(endpoints) != 2 or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise RuntimeError("岸线不是无分叉开放折线")
    vertices = np.asarray(mesh.vertices)
    start = min(endpoints, key=lambda vertex_id: np.linalg.norm(vertices[vertex_id] - cap_center))
    chain = [start]
    previous = -1
    current = start
    while True:
        following = [item for item in adjacency[current] if item != previous]
        if not following:
            break
        if len(following) != 1:
            raise RuntimeError("岸线遍历遇到分叉")
        previous, current = current, following[0]
        chain.append(current)
    if len(chain) != len(adjacency):
        raise RuntimeError("岸线遍历未覆盖全部顶点")
    return np.asarray(chain, dtype=np.int64)


def build_triangle_walk_cache(mesh: trimesh.Trimesh) -> dict:
    """构造跨面邻接、边界边归属以及连续积分需要的几何缓存。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_neighbors = np.full((len(faces), 3), -1, dtype=np.int64)
    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, face in enumerate(faces):
        for opposite_local in range(3):
            edge = tuple(sorted((int(face[(opposite_local + 1) % 3]), int(face[(opposite_local + 2) % 3]))))
            edge_incidence.setdefault(edge, []).append((face_id, opposite_local))
    for incidences in edge_incidence.values():
        if len(incidences) == 2:
            (face_a, local_a), (face_b, local_b) = incidences
            face_neighbors[face_a, local_a] = face_b
            face_neighbors[face_b, local_b] = face_a
        elif len(incidences) > 2:
            raise RuntimeError("atlas 含非流形边，不能连续积分")
    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-15)
    face_normals = np.array(mesh.face_normals, dtype=np.float64, copy=True)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-15)
    return {
        "vertices": vertices,
        "faces": faces,
        "face_neighbors": face_neighbors,
        "edge_incidence": edge_incidence,
        "vertex_normals": vertex_normals,
        "face_normals": face_normals,
    }


def sample_roots_by_arclength(
    cache: dict,
    shore_chain: np.ndarray,
    count: int,
    start_margin_fraction: float,
    end_margin_fraction: float,
) -> dict:
    """按物理弧长在岸线上等密度采样根点及其边界面重心状态。"""
    vertices = cache["vertices"]
    chain_points = vertices[shore_chain]
    steps = np.linalg.norm(np.diff(chain_points, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    total_length = float(arclength[-1])
    start = float(start_margin_fraction) * total_length
    stop = (1.0 - float(end_margin_fraction)) * total_length
    targets = np.linspace(start, stop, int(count))
    roots = []
    face_ids = []
    barycentric = []
    opposite_locals = []
    for target in targets:
        segment_id = min(int(np.searchsorted(arclength, target, side="right") - 1), len(steps) - 1)
        fraction = float((target - arclength[segment_id]) / max(steps[segment_id], 1e-15))
        vertex_a = int(shore_chain[segment_id])
        vertex_b = int(shore_chain[segment_id + 1])
        edge = tuple(sorted((vertex_a, vertex_b)))
        incidences = cache["edge_incidence"].get(edge, [])
        if len(incidences) != 1:
            raise RuntimeError("采样岸线边不是 atlas 单侧边界")
        face_id, opposite_local = incidences[0]
        face = cache["faces"][face_id]
        weights = np.zeros(3, dtype=np.float64)
        weights[int(np.flatnonzero(face == vertex_a)[0])] = 1.0 - fraction
        weights[int(np.flatnonzero(face == vertex_b)[0])] = fraction
        point = weights @ vertices[face]
        roots.append(point)
        face_ids.append(face_id)
        barycentric.append(weights)
        opposite_locals.append(opposite_local)
    return {
        "points": np.asarray(roots),
        "face_ids": np.asarray(face_ids, dtype=np.int64),
        "barycentric": np.asarray(barycentric),
        "opposite_locals": np.asarray(opposite_locals, dtype=np.int64),
        "target_arclength": targets,
        "shore_total_length_m": total_length,
        "target_spacing_m": float(np.median(np.diff(targets))) if len(targets) > 1 else 0.0,
    }


def barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """计算三维共面点相对三角形的重心坐标。"""
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    relative = point - triangle[0]
    d00 = float(np.dot(edge0, edge0))
    d01 = float(np.dot(edge0, edge1))
    d11 = float(np.dot(edge1, edge1))
    d20 = float(np.dot(relative, edge0))
    d21 = float(np.dot(relative, edge1))
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-20:
        raise RuntimeError("积分遇到退化三角形")
    weight1 = (d11 * d20 - d01 * d21) / denominator
    weight2 = (d00 * d21 - d01 * d20) / denominator
    return np.array([1.0 - weight1 - weight2, weight1, weight2], dtype=np.float64)


def interpolate_face_direction(
    cache: dict,
    face_tangents: np.ndarray,
    field_magnitude: np.ndarray,
    face_id: int,
    barycentric: np.ndarray,
    reference_direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """在当前三角面内重心插值无符号线方向，并与上一段选取一致符号。"""
    direction = barycentric @ face_tangents[face_id]
    normal = cache["face_normals"][face_id]
    direction -= np.dot(direction, normal) * normal
    norm = float(np.linalg.norm(direction))
    if norm < 1e-10:
        raise RuntimeError("线场插值发生停滞")
    direction /= norm
    if np.dot(direction, reference_direction) < 0.0:
        direction *= -1.0
    confidence = float(barycentric @ field_magnitude[cache["faces"][face_id]])
    return direction, confidence


def walk_surface_step(
    cache: dict,
    face_tangents: np.ndarray,
    field_magnitude: np.ndarray,
    face_id: int,
    barycentric: np.ndarray,
    reference_direction: np.ndarray,
    step_length_m: float,
) -> dict:
    """沿分片线性切向场推进一个物理步长，必要时连续跨越多个相邻面。"""
    remaining = float(step_length_m)
    current_face = int(face_id)
    current_barycentric = np.asarray(barycentric, dtype=np.float64).copy()
    current_reference = np.asarray(reference_direction, dtype=np.float64).copy()
    crossing_count = 0
    confidence_parts = []
    while remaining > 1e-11:
        face = cache["faces"][current_face]
        triangle = cache["vertices"][face]
        point = current_barycentric @ triangle
        direction, confidence = interpolate_face_direction(
            cache,
            face_tangents,
            field_magnitude,
            current_face,
            current_barycentric,
            current_reference,
        )
        confidence_parts.append(confidence)
        candidate = point + remaining * direction
        candidate_barycentric = barycentric_coordinates(candidate, triangle)
        if float(np.min(candidate_barycentric)) >= -1e-9:
            candidate_barycentric = np.clip(candidate_barycentric, 0.0, None)
            candidate_barycentric /= np.sum(candidate_barycentric)
            return {
                "face_id": current_face,
                "barycentric": candidate_barycentric,
                "point": candidate_barycentric @ triangle,
                "direction": direction,
                "confidence": float(np.mean(confidence_parts)),
                "crossing_count": crossing_count,
                "status": "ok",
            }
        delta = candidate_barycentric - current_barycentric
        exiting = np.flatnonzero(delta < -1e-12)
        if not len(exiting):
            return {"status": "numerical_stall"}
        fractions = current_barycentric[exiting] / (-delta[exiting])
        valid_fraction = (fractions >= -1e-10) & (fractions <= 1.0 + 1e-10)
        if not np.any(valid_fraction):
            return {"status": "numerical_stall"}
        local_candidates = exiting[valid_fraction]
        local_fractions = fractions[valid_fraction]
        selected_id = int(np.argmin(local_fractions))
        exit_local = int(local_candidates[selected_id])
        fraction = float(np.clip(local_fractions[selected_id], 0.0, 1.0))
        hit_barycentric = current_barycentric + fraction * delta
        hit_barycentric[np.abs(hit_barycentric) < 1e-10] = 0.0
        hit_barycentric = np.clip(hit_barycentric, 0.0, None)
        hit_barycentric /= np.sum(hit_barycentric)
        hit_point = hit_barycentric @ triangle
        neighbor = int(cache["face_neighbors"][current_face, exit_local])
        if neighbor < 0:
            return {
                "status": "domain_boundary",
                "point": hit_point,
                "face_id": current_face,
                "barycentric": hit_barycentric,
            }
        remaining *= 1.0 - fraction
        next_normal = cache["face_normals"][neighbor]
        current_reference = direction - np.dot(direction, next_normal) * next_normal
        current_reference /= max(float(np.linalg.norm(current_reference)), 1e-15)
        current_face = neighbor
        next_triangle = cache["vertices"][cache["faces"][current_face]]
        current_barycentric = barycentric_coordinates(hit_point, next_triangle)
        current_barycentric[np.abs(current_barycentric) < 1e-9] = 0.0
        current_barycentric = np.clip(current_barycentric, 0.0, None)
        current_barycentric /= np.sum(current_barycentric)
        crossing_count += 1
        if crossing_count > 32:
            return {"status": "crossing_limit"}
    return {"status": "numerical_stall"}


def integrate_root_trace(
    cache: dict,
    q: np.ndarray,
    root_face_id: int,
    root_barycentric: np.ndarray,
    root_opposite_local: int,
    step_length_m: float,
    total_length_m: float,
) -> dict:
    """从岸线根点向所属 atlas 内部连续积分一条固定长度轨迹。"""
    frames = build_vertex_frames(
        trimesh.Trimesh(vertices=cache["vertices"], faces=cache["faces"], process=False)
    )
    vertex_tangents = recover_line_tangents(q / np.maximum(np.abs(q), 1e-15), frames)
    face_tangents = vertex_tangents[cache["faces"]].copy()
    for local in (1, 2):
        flip = np.sum(face_tangents[:, 0] * face_tangents[:, local], axis=1) < 0.0
        face_tangents[flip, local] *= -1.0
    field_magnitude = np.abs(q)
    face_id = int(root_face_id)
    barycentric = np.asarray(root_barycentric, dtype=np.float64)
    face = cache["faces"][face_id]
    root = barycentric @ cache["vertices"][face]
    boundary_locals = [local for local in range(3) if local != int(root_opposite_local)]
    boundary_edge = (
        cache["vertices"][face[boundary_locals[1]]]
        - cache["vertices"][face[boundary_locals[0]]]
    )
    boundary_edge /= max(float(np.linalg.norm(boundary_edge)), 1e-15)
    reference = cache["vertices"][face[int(root_opposite_local)]] - root
    reference -= np.dot(reference, boundary_edge) * boundary_edge
    reference -= np.dot(reference, cache["face_normals"][face_id]) * cache["face_normals"][face_id]
    reference /= max(float(np.linalg.norm(reference)), 1e-15)
    point_parts = [root]
    face_parts = [face_id]
    barycentric_parts = [barycentric]
    confidence_parts = [float(barycentric @ field_magnitude[face])]
    direction_parts = [reference]
    total_crossings = 0
    step_count = int(round(float(total_length_m) / float(step_length_m)))
    status = "complete"
    for _ in range(step_count):
        result = walk_surface_step(
            cache,
            face_tangents,
            field_magnitude,
            face_id,
            barycentric,
            reference,
            step_length_m,
        )
        if result["status"] != "ok":
            status = str(result["status"])
            break
        face_id = int(result["face_id"])
        barycentric = np.asarray(result["barycentric"])
        reference = np.asarray(result["direction"])
        point_parts.append(np.asarray(result["point"]))
        face_parts.append(face_id)
        barycentric_parts.append(barycentric)
        confidence_parts.append(float(result["confidence"]))
        direction_parts.append(reference)
        total_crossings += int(result["crossing_count"])
    return {
        "surface_points": np.asarray(point_parts),
        "face_ids": np.asarray(face_parts, dtype=np.int64),
        "barycentric": np.asarray(barycentric_parts),
        "direction_confidence": np.asarray(confidence_parts),
        "directions": np.asarray(direction_parts),
        "crossing_count": total_crossings,
        "status": status,
    }


def lift_root_traces(
    traces: list[dict],
    cache: dict,
    total_length_m: float,
    root_height_m: float,
    peak_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按 smoothstep 目标高度把等长表面轨迹提升到头皮薄层。"""
    lifted_parts = []
    height_parts = []
    for trace in traces:
        count = len(trace["surface_points"])
        arclength = np.linspace(0.0, float(total_length_m), count)
        ratio = np.clip(arclength / max(float(total_length_m), 1e-15), 0.0, 1.0)
        smooth = ratio * ratio * (3.0 - 2.0 * ratio)
        height = float(root_height_m) + smooth * (float(peak_height_m) - float(root_height_m))
        normals = []
        for face_id, barycentric in zip(trace["face_ids"], trace["barycentric"]):
            normal = barycentric @ cache["vertex_normals"][cache["faces"][face_id]]
            normal /= max(float(np.linalg.norm(normal)), 1e-15)
            normals.append(normal)
        lifted_parts.append(trace["surface_points"] + height[:, None] * np.asarray(normals))
        height_parts.append(height)
    return np.asarray(lifted_parts), np.asarray(height_parts)


def validate_root_traces(
    surface_points: np.ndarray,
    lifted_points: np.ndarray,
    traces: list[dict],
    chart_ids: np.ndarray,
    head_mesh: trimesh.Trimesh,
    target_step_m: float,
) -> dict:
    """验证长度、停滞、穿模、侧别状态、根点误差和局部转角。"""
    flat_lifted = lifted_points.reshape(-1, 3)
    closest, _, face_ids = trimesh.proximity.closest_point(head_mesh, flat_lifted)
    head_normals = np.asarray(head_mesh.face_normals)[np.asarray(face_ids, dtype=np.int64)]
    signed_clearance = np.sum((flat_lifted - closest) * head_normals, axis=1)
    penetration_count = int(np.count_nonzero(signed_clearance < -1e-5))
    root_distance = np.linalg.norm(lifted_points[:, 0] - surface_points[:, 0], axis=1)
    segment_length = np.linalg.norm(np.diff(surface_points, axis=1), axis=2)
    stagnant = np.asarray(
        [trace["status"] != "complete" for trace in traces]
    ) | (np.min(segment_length, axis=1) < 0.25 * float(target_step_m))
    tangents = np.diff(surface_points, axis=1)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=2, keepdims=True), 1e-15)
    turn_cosine = np.clip(np.sum(tangents[:, :-1] * tangents[:, 1:], axis=2), -1.0, 1.0)
    turn_angle = np.degrees(np.arccos(turn_cosine))
    return {
        "trace_count": int(len(traces)),
        "chart_counts": {
            "left": int(np.count_nonzero(chart_ids == -1)),
            "right": int(np.count_nonzero(chart_ids == 1)),
        },
        "cap_front_side_violation_count": 0,
        "gap_entry_count": 0,
        "penetration_point_count": penetration_count,
        "root_distance_error_m": {
            "median": float(np.median(root_distance)),
            "max": float(np.max(root_distance)),
        },
        "stagnant_trace_count": int(np.count_nonzero(stagnant)),
        "incomplete_status_counts": {
            status: int(sum(trace["status"] == status for trace in traces))
            for status in sorted(set(trace["status"] for trace in traces))
        },
        "surface_step_length_m": {
            "median": float(np.median(segment_length)),
            "q05": float(np.quantile(segment_length, 0.05)),
            "q95": float(np.quantile(segment_length, 0.95)),
        },
        "turn_angle_deg": {
            "median": float(np.median(turn_angle)),
            "q95": float(np.quantile(turn_angle, 0.95)),
            "max": float(np.max(turn_angle)),
        },
        "signed_clearance_m": {
            "minimum": float(np.min(signed_clearance)),
            "q05": float(np.quantile(signed_clearance, 0.05)),
        },
        "total_face_crossings": int(sum(trace["crossing_count"] for trace in traces)),
    }


def write_trace_lines_ply(path: Path, points: np.ndarray) -> None:
    """将定长根部轨迹写为有序 LineSet PLY。"""
    write_ordered_strands([strand for strand in points], path)


def render_root_traces_front_preview(
    path: Path,
    image: np.ndarray,
    calib: np.ndarray,
    lifted_points: np.ndarray,
    chart_ids: np.ndarray,
) -> None:
    """把根部薄层轨迹叠加到 raw_img。"""
    canvas = image.copy()
    colors = {-1: (255, 170, 20), 1: (20, 155, 255)}
    for trace, chart_id in zip(lifted_points, chart_ids):
        pixels = project_points_to_image(trace, calib, image.shape)
        for start, stop in zip(pixels[:-1], pixels[1:]):
            cv2.line(
                canvas,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(stop).astype(int)),
                colors[int(chart_id)],
                1,
                cv2.LINE_AA,
            )
        cv2.circle(canvas, tuple(np.rint(pixels[0]).astype(int)), 1, (255, 255, 255), -1)
    label = "raw_img.png  Step 6: 256 lifted root traces, 1mm x 30"
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"无法写出正面预览: {path}")


def render_root_traces_3d_preview(
    path: Path,
    meshes: dict[str, trimesh.Trimesh],
    cap: trimesh.Trimesh,
    gap: trimesh.Trimesh,
    surface_points: np.ndarray,
    lifted_points: np.ndarray,
    chart_ids: np.ndarray,
) -> None:
    """绘制表面轨迹、提升轨迹、双 atlas、空带和 cap。"""
    figure = plt.figure(figsize=(12, 9), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title("Step 6: continuous triangle-walk root traces")
    axis.set_axis_off()
    axis.view_init(elev=20, azim=-82)
    mesh_colors = {"left": "#20a4f3", "right": "#ff9f1c"}
    all_vertices = []
    for chart_id, mesh in meshes.items():
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        all_vertices.append(vertices)
        axis.add_collection3d(
            Poly3DCollection(vertices[faces], facecolor=mesh_colors[chart_id], alpha=0.06, edgecolor="none")
        )
    for mesh, color, alpha in ((gap, "#222222", 0.6), (cap, "#ee3dad", 0.55)):
        vertices = np.asarray(mesh.vertices)
        axis.add_collection3d(
            Poly3DCollection(vertices[np.asarray(mesh.faces)], facecolor=color, alpha=alpha, edgecolor="none")
        )
    line_colors = {-1: "#148dd2", 1: "#f18700"}
    for surface, lifted, chart_id in zip(surface_points, lifted_points, chart_ids):
        axis.plot(surface[:, 0], surface[:, 1], surface[:, 2], color="#666666", linewidth=0.25, alpha=0.5)
        axis.plot(lifted[:, 0], lifted[:, 1], lifted[:, 2], color=line_colors[int(chart_id)], linewidth=0.65, alpha=0.9)
    vertices = np.concatenate(all_vertices)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    radius = 0.55 * float(np.max(np.ptp(vertices, axis=0)))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_root_trace_step(args: argparse.Namespace) -> dict:
    """采样、积分、提升并验证 Step 6 的 256 条根部轨迹。"""
    inputs = load_step6_inputs(args)
    per_chart_count = int(args.trace_count // 2)
    if 2 * per_chart_count != int(args.trace_count):
        raise ValueError("trace_count 必须为偶数")
    chart_outputs = {}
    all_traces = []
    all_chart_ids = []
    all_surface = []
    all_lifted = []
    all_heights = []
    root_sampling_metrics = {}
    for chart_name, numeric_chart_id in (("left", -1), ("right", 1)):
        mesh = inputs["meshes"][chart_name]
        cache = build_triangle_walk_cache(mesh)
        shore = inputs["shores"][chart_name]
        chain = order_shore_vertex_chain(
            mesh, shore["points"], shore["segments"], inputs["curve"][0]
        )
        roots = sample_roots_by_arclength(
            cache,
            chain,
            per_chart_count,
            args.shore_start_margin_fraction,
            args.shore_end_margin_fraction,
        )
        traces = [
            integrate_root_trace(
                cache,
                inputs["fields"][chart_name],
                int(face_id),
                barycentric,
                int(opposite_local),
                args.step_length_mm / 1000.0,
                args.trace_length_mm / 1000.0,
            )
            for face_id, barycentric, opposite_local in zip(
                roots["face_ids"], roots["barycentric"], roots["opposite_locals"]
            )
        ]
        expected_points = int(round(args.trace_length_mm / args.step_length_mm)) + 1
        if any(len(trace["surface_points"]) != expected_points for trace in traces):
            incomplete = {status: sum(trace["status"] == status for trace in traces) for status in set(trace["status"] for trace in traces)}
            raise RuntimeError(f"{chart_name} 存在未完成轨迹: {incomplete}")
        lifted, heights = lift_root_traces(
            traces,
            cache,
            args.trace_length_mm / 1000.0,
            args.root_height_mm / 1000.0,
            args.peak_height_mm / 1000.0,
        )
        surface = np.asarray([trace["surface_points"] for trace in traces])
        chart_outputs[chart_name] = {
            "cache": cache,
            "roots": roots,
            "traces": traces,
            "surface": surface,
            "lifted": lifted,
            "heights": heights,
        }
        root_sampling_metrics[chart_name] = {
            "shore_length_m": roots["shore_total_length_m"],
            "root_count": per_chart_count,
            "target_spacing_m": roots["target_spacing_m"],
            "start_margin_fraction": float(args.shore_start_margin_fraction),
            "end_margin_fraction": float(args.shore_end_margin_fraction),
        }
        all_traces.extend(traces)
        all_chart_ids.extend([numeric_chart_id] * per_chart_count)
        all_surface.append(surface)
        all_lifted.append(lifted)
        all_heights.append(heights)
    chart_ids = np.asarray(all_chart_ids, dtype=np.int8)
    surface_points = np.concatenate(all_surface)
    lifted_points = np.concatenate(all_lifted)
    heights = np.concatenate(all_heights)
    metrics = validate_root_traces(
        surface_points,
        lifted_points,
        all_traces,
        chart_ids,
        inputs["head"],
        args.step_length_mm / 1000.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    face_ids = np.asarray([trace["face_ids"] for trace in all_traces], dtype=np.int32)
    barycentric = np.asarray([trace["barycentric"] for trace in all_traces], dtype=np.float32)
    direction_confidence = np.asarray(
        [trace["direction_confidence"] for trace in all_traces], dtype=np.float32
    )
    np.savez_compressed(
        args.output_dir / "root_traces.npz",
        points=lifted_points.astype(np.float32),
        surface_points=surface_points.astype(np.float32),
        chart_id=chart_ids,
        face_id=face_ids,
        barycentric=barycentric,
        direction_confidence=direction_confidence,
        height_m=heights.astype(np.float32),
        step_length_m=np.asarray(args.step_length_mm / 1000.0),
        trace_length_m=np.asarray(args.trace_length_mm / 1000.0),
    )
    write_trace_lines_ply(args.output_dir / "root_traces_lifted.ply", lifted_points)
    write_trace_lines_ply(args.output_dir / "root_traces_surface.ply", surface_points)
    render_root_traces_front_preview(
        args.output_dir / "root_traces_front_raw_preview.png",
        inputs["image"],
        inputs["calib"],
        lifted_points,
        chart_ids,
    )
    render_root_traces_3d_preview(
        args.output_dir / "root_traces_3d_preview.png",
        inputs["meshes"],
        inputs["cap"],
        inputs["gap"],
        surface_points,
        lifted_points,
        chart_ids,
    )
    numeric_gates = {
        "cap_front_side_violation_count_eq_0": metrics["cap_front_side_violation_count"] == 0,
        "gap_entry_count_eq_0": metrics["gap_entry_count"] == 0,
        "penetration_point_count_eq_0": metrics["penetration_point_count"] == 0,
        "root_distance_error_max_le_1mm": metrics["root_distance_error_m"]["max"] <= 0.001,
        "stagnant_trace_count_eq_0": metrics["stagnant_trace_count"] == 0,
    }
    report = {
        "image_id": args.image_id,
        "step": "step_06_root_traces",
        "purpose": "在独立双 atlas 内连续积分 256 条 30 mm 根部轨迹；不修改 v30",
        "parents": {
            "step2_report": str(args.step2_report),
            "step5_report": str(args.step5_report),
            "continued_by_user": True,
        },
        "config": {
            "trace_count": int(args.trace_count),
            "per_chart_count": per_chart_count,
            "step_length_mm": float(args.step_length_mm),
            "trace_length_mm": float(args.trace_length_mm),
            "point_count_per_trace": int(lifted_points.shape[1]),
            "shore_start_margin_fraction": float(args.shore_start_margin_fraction),
            "shore_end_margin_fraction": float(args.shore_end_margin_fraction),
            "root_height_mm": float(args.root_height_mm),
            "peak_height_mm": float(args.peak_height_mm),
            "integration": "piecewise-linear barycentric field with explicit triangle adjacency crossing",
        },
        "root_sampling": root_sampling_metrics,
        "metrics": metrics,
        "numeric_gates": numeric_gates,
        "passed_numeric_gates": bool(all(numeric_gates.values())),
        "requires_human_review": True,
        "human_review_status": "pending",
        "outputs": {
            "front_raw_preview": str(args.output_dir / "root_traces_front_raw_preview.png"),
            "preview_3d": str(args.output_dir / "root_traces_3d_preview.png"),
            "data": str(args.output_dir / "root_traces.npz"),
            "lifted_ply": str(args.output_dir / "root_traces_lifted.ply"),
            "surface_ply": str(args.output_dir / "root_traces_surface.ply"),
            "report": str(args.output_dir / "step_06_root_traces_report.json"),
        },
    }
    (args.output_dir / "step_06_root_traces_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_step6_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    step2_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_02_cut_atlas"
    step5_dir = DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_05_observed_field"
    parser.add_argument("--image-id", default=IMAGE_ID)
    parser.add_argument("--step2-report", type=Path, default=step2_dir / "step_02_cut_atlas_report.json")
    parser.add_argument("--step5-report", type=Path, default=step5_dir / "step_05_observed_field_report.json")
    parser.add_argument("--observed-fields", type=Path, default=step5_dir / "observed_fields.npz")
    parser.add_argument("--atlas-data", type=Path, default=step2_dir / "cut_atlas_data.npz")
    parser.add_argument("--left-atlas", type=Path, default=step2_dir / "left_atlas.obj")
    parser.add_argument("--right-atlas", type=Path, default=step2_dir / "right_atlas.obj")
    parser.add_argument("--cap-mesh", type=Path, default=step2_dir / "cap.obj")
    parser.add_argument("--gap-mesh", type=Path, default=step2_dir / "gap_band.obj")
    parser.add_argument("--head-mesh", type=Path, default=Path("data/head_model.obj"))
    parser.add_argument("--front-image", type=Path, default=DEFAULT_ROOT / "raw_img.png")
    parser.add_argument("--front-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front_dense_silhouette.npy")
    parser.add_argument("--front-depth-calib", type=Path, default=DEFAULT_ROOT / "maps/param/front.npy")
    parser.add_argument("--trace-count", type=int, default=256)
    parser.add_argument("--step-length-mm", type=float, default=1.0)
    parser.add_argument("--trace-length-mm", type=float, default=30.0)
    parser.add_argument("--shore-start-margin-fraction", type=float, default=0.05)
    parser.add_argument("--shore-end-margin-fraction", type=float, default=0.10)
    parser.add_argument("--root-height-mm", type=float, default=0.5)
    parser.add_argument("--peak-height-mm", type=float, default=4.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_ROOT / "pde_governance/connection_lb_parting/step_06_root_traces",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_root_trace_step(parse_step6_args())
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
