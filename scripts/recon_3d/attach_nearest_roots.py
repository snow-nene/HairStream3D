"""对重建 PLY 做最终发根吸附，不依赖 PDE 或发缝 mask。"""
import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
import trimesh


def _load_head_mesh(head_mesh):
    if isinstance(head_mesh, trimesh.Trimesh):
        return head_mesh
    if str(head_mesh).lower().endswith(".npz"):
        data = np.load(head_mesh)
        return trimesh.Trimesh(
            vertices=data["vertices"], faces=data["faces"], process=False
        )
    return trimesh.load(head_mesh, process=False)


def _transform_strands_to_blender(strands):
    transformed = []
    for strand in strands:
        strand = np.asarray(strand, dtype=np.float32)
        transformed.append(
            np.column_stack([strand[:, 0], -strand[:, 2], strand[:, 1]])
            .astype(np.float32)
        )
    return transformed


def _transform_strands_from_blender(strands):
    transformed = []
    for strand in strands:
        strand = np.asarray(strand, dtype=np.float32)
        transformed.append(
            np.column_stack([strand[:, 0], strand[:, 2], -strand[:, 1]])
            .astype(np.float32)
        )
    return transformed


def read_ordered_strands(ply_path):
    """按 PLY 线段拓扑恢复有序发丝，兼容冻结后重复的尾点。"""
    line_set = o3d.io.read_line_set(ply_path)
    points = np.asarray(line_set.points, dtype=np.float32)
    lines = np.asarray(line_set.lines, dtype=np.int64)
    if len(points) == 0 or len(lines) == 0:
        raise ValueError(f"PLY 不包含有效折线: {ply_path}")

    strands = []
    current = [int(lines[0, 0]), int(lines[0, 1])]
    for previous, edge in zip(lines[:-1], lines[1:]):
        if int(edge[0]) == int(previous[1]):
            current.append(int(edge[1]))
        else:
            strands.append(points[np.asarray(current, dtype=np.int64)].copy())
            current = [int(edge[0]), int(edge[1])]
    strands.append(points[np.asarray(current, dtype=np.int64)].copy())
    return strands


def write_ordered_strands(strands, output_path, drop_collapsed=False):
    """保持每根折线的点数与顺序写回 Open3D LineSet。"""
    points = []
    lines = []
    offset = 0
    for strand in strands:
        strand = np.asarray(strand, dtype=np.float32)
        if drop_collapsed and _effective_last_index(strand) == 0:
            continue
        points.append(strand)
        lines.extend([[offset + index, offset + index + 1]
                      for index in range(len(strand) - 1)])
        offset += len(strand)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.concatenate(points, axis=0)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not o3d.io.write_line_set(output_path, line_set):
        raise RuntimeError(f"无法写出 PLY: {output_path}")


def _effective_last_index(strand):
    moving = np.linalg.norm(np.diff(strand, axis=0), axis=1) > 1e-7
    ids = np.flatnonzero(moving)
    return int(ids[-1] + 1) if len(ids) else 0


def _closest_points_chunked(mesh, points, chunk_size=50000):
    closest_parts = []
    distance_parts = []
    face_parts = []
    for start in range(0, len(points), int(chunk_size)):
        stop = min(start + int(chunk_size), len(points))
        closest, distance, face_ids = trimesh.proximity.closest_point(
            mesh, points[start:stop]
        )
        closest_parts.append(closest)
        distance_parts.append(distance)
        face_parts.append(face_ids)
    return (
        np.concatenate(closest_parts),
        np.concatenate(distance_parts),
        np.concatenate(face_parts),
    )


def _load_projection_calibration(calib_path, image_size=1024):
    """读取 HairStep 正交相机参数并构造 NDC 投影矩阵。"""
    param = np.load(calib_path, allow_pickle=True).item()
    ortho_ratio = float(param["ortho_ratio"])
    scale = float(np.asarray(param["scale"]).reshape(-1)[0])
    center = np.asarray(param["center"], dtype=np.float64).reshape(3, 1)
    rotation = np.asarray(param["R"], dtype=np.float64)
    translate = -rotation @ center
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3] = np.concatenate([rotation, translate], axis=1)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = scale / ortho_ratio / (float(image_size) / 2.0)
    intrinsic[1, 1] = -scale / ortho_ratio / (float(image_size) / 2.0)
    intrinsic[2, 2] = scale / ortho_ratio / (float(image_size) / 2.0)
    return intrinsic @ extrinsic


def _project_pixels(points, calib, image_shape):
    height, width = image_shape
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    clip = homogeneous @ np.asarray(calib, dtype=np.float64).T
    ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
    return np.column_stack([
        (ndc[:, 0] + 1.0) * 0.5 * (width - 1),
        (ndc[:, 1] + 1.0) * 0.5 * (height - 1),
    ])


def select_roots_near_parting(strands, parting_mask, calib, radius_px=48):
    """选择发根投影落在发缝邻域内的发丝，限制包络修正的影响范围。"""
    mask = np.asarray(parting_mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    roots = np.stack([np.asarray(strand)[0] for strand in strands])
    pixels = _project_pixels(roots, calib, mask.shape)
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    selected = np.zeros(len(strands), dtype=bool)
    selected[inside] = mask[y[inside], x[inside]]
    return selected


def enforce_parting_safe_roots(
    strands,
    head_mesh,
    parting_mask,
    calib,
    head_space="reconstruction",
    boundary_offset_px=4.0,
    transition_steps=3,
    surface_distance=-0.0015,
    max_shift_distance=0.03,
):
    """吸附完成后，把仍落在发缝内的根局部移回同侧最近头皮。"""
    out = [np.asarray(strand, dtype=np.float32).copy() for strand in strands]
    mask = np.asarray(parting_mask, dtype=bool)
    height, width = mask.shape
    mask_y, mask_x = np.where(mask)
    if len(mask_x) == 0 or not out:
        return out, {"violating_roots_before": 0, "corrected_roots": 0}

    left = np.full(height, np.nan, dtype=np.float64)
    right = np.full(height, np.nan, dtype=np.float64)
    for row in np.unique(mask_y):
        row_x = mask_x[mask_y == row]
        left[row] = float(np.min(row_x))
        right[row] = float(np.max(row_x))
    valid_rows = np.flatnonzero(np.isfinite(left))
    left = np.interp(np.arange(height), valid_rows, left[valid_rows])
    right = np.interp(np.arange(height), valid_rows, right[valid_rows])

    roots = np.stack([strand[0] for strand in out]).astype(np.float64)
    pixels = _project_pixels(roots, calib, mask.shape)
    px = pixels[:, 0]
    py = pixels[:, 1]
    x = np.rint(px).astype(np.int64)
    y = np.rint(py).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    violating = np.zeros(len(out), dtype=bool)
    violating[inside] = mask[y[inside], x[inside]]
    ids = np.flatnonzero(violating)
    if len(ids) == 0:
        return out, {"violating_roots_before": 0, "corrected_roots": 0}

    rows = np.rint(py[ids]).astype(np.int64).clip(0, height - 1)
    center = 0.5 * (left[rows] + right[rows])
    side = np.where(px[ids] < center, -1.0, 1.0)
    target_px = np.where(
        side < 0.0,
        left[rows] - float(boundary_offset_px),
        right[rows] + float(boundary_offset_px),
    )
    matrix = np.asarray(calib, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    homogeneous = np.column_stack([roots[ids], np.ones(len(ids))])
    clip = homogeneous @ matrix.T
    target_ndc_x = target_px / max(width - 1, 1) * 2.0 - 1.0
    clip[:, 0] = target_ndc_x * clip[:, 3]
    provisional_h = clip @ inverse.T
    provisional = provisional_h[:, :3] / (provisional_h[:, 3:4] + 1e-12)

    mesh = _load_head_mesh(head_mesh)
    if head_space == "blender":
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        vertices = np.column_stack([
            vertices[:, 0], vertices[:, 2], -vertices[:, 1]
        ])
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=np.asarray(mesh.faces), process=False
        )
    closest, _, face_ids = _closest_points_chunked(mesh, provisional)
    normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)].copy()
    inward = np.sum(normals * (closest - mesh.centroid), axis=1) < 0.0
    normals[inward] *= -1.0
    corrected = closest + normals * float(surface_distance)
    shift = np.linalg.norm(corrected - roots[ids], axis=1)
    accepted = np.all(np.isfinite(corrected), axis=1) & (
        shift <= float(max_shift_distance)
    )
    accepted_ids = ids[accepted]
    steps = max(1, int(transition_steps))
    for owner, root in zip(accepted_ids, corrected[accepted]):
        count = min(steps, len(out[owner]))
        delta = root.astype(np.float32) - out[owner][0]
        weights = np.linspace(1.0, 0.0, count, dtype=np.float32)
        out[owner][:count] += delta[None, :] * weights[:, None]
        out[owner][0] = root.astype(np.float32)

    remaining = select_roots_near_parting(out, mask, calib, radius_px=0)
    accepted_shift = shift[accepted]
    report = {
        "violating_roots_before": int(len(ids)),
        "corrected_roots": int(np.sum(accepted)),
        "rejected_large_shift": int(np.sum(~accepted)),
        "remaining_roots_in_parting": int(np.sum(remaining)),
        "boundary_offset_px": float(boundary_offset_px),
        "surface_distance_m": float(surface_distance),
        "max_shift_distance_m": float(max_shift_distance),
        "accepted_shift_mm": {
            "median": float(np.median(accepted_shift) * 1000.0)
            if len(accepted_shift) else 0.0,
            "max": float(np.max(accepted_shift) * 1000.0)
            if len(accepted_shift) else 0.0,
        },
    }
    return out, report


def enforce_root_clearance_envelope(
    strands,
    head_mesh,
    selected=None,
    contact_length=0.012,
    release_length=0.042,
    contact_clearance=0.0002,
    release_clearance=0.009,
):
    """按根部弧长限制头皮间距，避免固定点数投影造成突跳或压平。"""
    mesh = _load_head_mesh(head_mesh)
    out = [np.asarray(strand, dtype=np.float32).copy() for strand in strands]
    if selected is None:
        selected = np.ones(len(out), dtype=bool)
    else:
        selected = np.asarray(selected, dtype=bool).reshape(-1)
    if len(selected) != len(out):
        raise ValueError("selected 必须与发丝数量一致")

    samples = []
    owners = []
    indices = []
    caps = []
    release_span = max(float(release_length) - float(contact_length), 1e-8)
    for owner, strand in enumerate(out):
        if not selected[owner]:
            continue
        last = _effective_last_index(strand)
        if last < 1:
            continue
        active = strand[:last + 1].astype(np.float64)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(active, axis=0), axis=1))])
        local_ids = np.flatnonzero(arc <= float(release_length))
        if len(local_ids) == 0:
            continue
        t = np.clip((arc[local_ids] - float(contact_length)) / release_span, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        allowed = float(contact_clearance) + smooth * (
            float(release_clearance) - float(contact_clearance)
        )
        samples.append(active[local_ids])
        owners.extend([owner] * len(local_ids))
        indices.extend(local_ids.tolist())
        caps.extend(allowed.tolist())
    if not samples:
        return out, {"selected_strands": 0, "adjusted_points": 0}

    points = np.concatenate(samples).astype(np.float64)
    closest, distances, face_ids = _closest_points_chunked(mesh, points)
    normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)].copy()
    inward = np.sum(normals * (points - closest), axis=1) < 0.0
    normals[inward] *= -1.0
    caps = np.asarray(caps, dtype=np.float64)
    adjust = distances > caps
    targets = closest + normals * caps[:, None]
    for owner, index, target, should_adjust in zip(owners, indices, targets, adjust):
        if should_adjust:
            out[owner][index] = target.astype(np.float32)
    return out, {
        "selected_strands": int(np.sum(selected)),
        "audited_points": int(len(points)),
        "adjusted_points": int(np.sum(adjust)),
        "contact_length_m": float(contact_length),
        "release_length_m": float(release_length),
        "contact_clearance_m": float(contact_clearance),
        "release_clearance_m": float(release_clearance),
        "distance_before_mm": {
            "median": float(np.median(distances) * 1000.0),
            "p95": float(np.percentile(distances, 95) * 1000.0),
            "max": float(np.max(distances) * 1000.0),
        },
    }


def filter_floating_parting_segments(
    strands,
    head_mesh,
    parting_mask,
    calib,
    corridor_dilation_px=6,
    clearance=0.003,
    protected_root_steps=8,
    head_space="blender",
):
    """剔除从中段或发尾悬空进入真实发缝区域的拓扑异常发丝。"""
    out = [np.asarray(strand, dtype=np.float32).copy() for strand in strands]
    mask = np.asarray(parting_mask, dtype=bool)
    dilation = max(0, int(corridor_dilation_px))
    if dilation:
        size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    mesh = _load_head_mesh(head_mesh)

    points = []
    owners = []
    steps = []
    last_indices = np.asarray([_effective_last_index(strand) for strand in out])
    for owner, (strand, last) in enumerate(zip(out, last_indices)):
        if last < 1:
            continue
        active = strand[:last + 1]
        points.append(active)
        owners.extend([owner] * len(active))
        steps.extend(range(len(active)))
    if not points:
        return out, {"rejected_strands": 0}
    points = np.concatenate(points).astype(np.float64)
    owners = np.asarray(owners, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    pixels = _project_pixels(points, calib, mask.shape)
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    in_corridor = np.zeros(len(points), dtype=bool)
    in_corridor[inside] = mask[y[inside], x[inside]]

    distance_points = (
        np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
        if head_space == "blender" else points
    )
    _, distances, _ = _closest_points_chunked(mesh, distance_points)
    floating = in_corridor & (distances > float(clearance))
    rejected = np.unique(owners[floating & (steps >= int(protected_root_steps))])
    tip_count = 0
    middle_count = 0
    rejected_set = set(int(owner) for owner in rejected)
    for owner in rejected:
        bad_steps = steps[(owners == owner) & floating]
        first = int(np.min(bad_steps))
        if int(last_indices[owner]) - first <= int(protected_root_steps):
            tip_count += 1
        else:
            middle_count += 1
    filtered = [strand for owner, strand in enumerate(out) if owner not in rejected_set]
    return filtered, {
        "rejected_strands": int(len(rejected)),
        "middle_segment_strands": int(middle_count),
        "tip_segment_strands": int(tip_count),
        "corridor_dilation_px": int(dilation),
        "clearance_m": float(clearance),
        "protected_root_steps": int(protected_root_steps),
    }


def attach_nearest_roots(
    strands,
    head_mesh,
    attach_steps=10,
    hard_steps=5,
    probe_steps=5,
    target_distance=0.0,
    max_root_distance=0.0,
    endpoint_mode="start",
):
    """把发根段吸附到最近表面，可选为未知拓扑自动判断端点。"""
    if attach_steps <= 0 or hard_steps <= 0:
        return [np.asarray(strand).copy() for strand in strands], {}
    mesh = _load_head_mesh(head_mesh)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("head_mesh 必须是包含三角面的 Trimesh 或模型路径")

    out = [np.asarray(strand, dtype=np.float32).copy() for strand in strands]
    last_indices = np.asarray([_effective_last_index(strand) for strand in out])
    valid = last_indices > 0
    probe = max(1, int(probe_steps))
    reversed_count = 0
    root_distances = np.full(len(out), np.inf, dtype=np.float64)
    valid_ids = np.flatnonzero(valid)
    if endpoint_mode == "nearest":
        endpoint_samples = []
        endpoint_owners = []
        for index in valid_ids:
            strand = out[index]
            last = last_indices[index]
            active = strand[:last + 1]
            count = min(probe, len(active))
            endpoint_samples.extend([active[:count], active[-count:][::-1]])
            endpoint_owners.append(index)
        flat_probe = np.concatenate(endpoint_samples, axis=0).astype(np.float64)
        _, probe_distances, _ = _closest_points_chunked(mesh, flat_probe)
        cursor = 0
        for owner in endpoint_owners:
            count = min(probe, last_indices[owner] + 1)
            start_score = float(np.median(probe_distances[cursor:cursor + count]))
            cursor += count
            end_score = float(np.median(probe_distances[cursor:cursor + count]))
            cursor += count
            root_distances[owner] = min(start_score, end_score)
            if end_score < start_score:
                last = int(last_indices[owner])
                active = out[owner][:last + 1][::-1].copy()
                out[owner][:last + 1] = active
                out[owner][last + 1:] = active[-1]
                reversed_count += 1
    elif endpoint_mode == "start":
        roots = np.stack([out[index][0] for index in valid_ids]).astype(np.float64)
        _, distances, _ = _closest_points_chunked(mesh, roots)
        root_distances[valid_ids] = distances
    else:
        raise ValueError(f"未知 endpoint_mode: {endpoint_mode}")

    selected = valid.copy()
    if max_root_distance > 0.0:
        selected &= root_distances <= float(max_root_distance)
    selected_ids = np.flatnonzero(selected)
    max_steps = min(
        int(attach_steps),
        min(len(out[index]) for index in selected_ids),
    )
    hard = min(int(hard_steps), max_steps)
    original = np.stack(
        [out[index][:max_steps] for index in selected_ids], axis=0
    ).astype(np.float64)
    flat = original.reshape(-1, 3)
    closest, before_distance, face_ids = _closest_points_chunked(mesh, flat)
    normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)].copy()
    inward = np.sum(normals * (flat - closest), axis=1) < 0.0
    normals[inward] *= -1.0
    shell = (
        closest + normals * float(target_distance)
    ).reshape(original.shape)

    weights = np.ones(max_steps, dtype=np.float64)
    if max_steps > hard:
        weights[hard:] = np.linspace(
            1.0, 0.0, max_steps - hard + 1, dtype=np.float64
        )[1:]
    attached = original * (1.0 - weights[None, :, None]) + shell * weights[None, :, None]
    for index, values in zip(selected_ids, attached):
        out[index][:max_steps] = values.astype(np.float32)

    _, after_distance, _ = _closest_points_chunked(
        mesh, attached[:, :hard].reshape(-1, 3)
    )
    report = {
        "strand_count": len(out),
        "valid_strands": int(valid.sum()),
        "attached_strands": int(len(selected_ids)),
        "reversed_strands": int(reversed_count),
        "endpoint_mode": endpoint_mode,
        "attach_steps": int(max_steps),
        "hard_steps": int(hard),
        "target_distance_m": float(target_distance),
        "root_probe_distance_mm": {
            "median": float(np.median(root_distances[valid]) * 1000.0),
            "p95": float(np.percentile(root_distances[valid], 95) * 1000.0),
            "max": float(np.max(root_distances[valid]) * 1000.0),
        },
        "hard_point_distance_after_mm": {
            "median": float(np.median(after_distance) * 1000.0),
            "p95": float(np.percentile(after_distance, 95) * 1000.0),
            "max": float(np.max(after_distance) * 1000.0),
        },
    }
    return out, report


def main():
    parser = argparse.ArgumentParser(description="自动识别并吸附 PLY 发根")
    parser.add_argument("--input", required=True, help="输入发丝 PLY")
    parser.add_argument("--head_mesh", default="data/head_model.obj")
    parser.add_argument("--output", required=True, help="输出吸附后 PLY")
    parser.add_argument("--report", default=None, help="可选 JSON 诊断输出")
    parser.add_argument("--attach_steps", type=int, default=10)
    parser.add_argument("--hard_steps", type=int, default=5)
    parser.add_argument("--probe_steps", type=int, default=5)
    parser.add_argument("--target_distance", type=float, default=0.0)
    parser.add_argument(
        "--endpoint_mode", choices=("start", "nearest"), default="start",
        help="PDE PLY 使用 start；未知外部 PLY 可用 nearest 自动判断端点",
    )
    parser.add_argument(
        "--head_space", choices=("reconstruction", "blender"),
        default="reconstruction",
        help="头模所在坐标系；blender 会自动转换 PLY 坐标",
    )
    parser.add_argument(
        "--max_root_distance", type=float, default=0.0,
        help="最大自动吸附距离（米）；<=0 表示不限制",
    )
    parser.add_argument("--parting_mask", default=None, help="可选发缝区域 mask")
    parser.add_argument("--front_calib", default=None, help="可选 front 正交相机参数")
    parser.add_argument("--parting_envelope_radius_px", type=int, default=48)
    parser.add_argument(
        "--enforce_parting_safe_roots", action="store_true",
        help="最近头皮吸附后，把仍落在发缝 mask 内的根局部移回同侧头皮",
    )
    parser.add_argument("--parting_safe_boundary_offset_px", type=float, default=4.0)
    parser.add_argument("--parting_safe_transition_steps", type=int, default=3)
    parser.add_argument("--parting_safe_max_shift", type=float, default=0.03)
    parser.add_argument(
        "--parting_safe_surface_distance", type=float, default=-0.0015
    )
    parser.add_argument("--envelope_contact_length", type=float, default=0.0)
    parser.add_argument("--envelope_release_length", type=float, default=0.0)
    parser.add_argument("--envelope_contact_clearance", type=float, default=0.0002)
    parser.add_argument("--envelope_release_clearance", type=float, default=0.009)
    parser.add_argument("--filter_parting_topology", action="store_true")
    parser.add_argument("--parting_corridor_dilation_px", type=int, default=6)
    parser.add_argument("--parting_floating_clearance", type=float, default=0.003)
    parser.add_argument("--parting_protected_root_steps", type=int, default=8)
    parser.add_argument(
        "--drop_collapsed_strands", action="store_true",
        help="写出时彻底移除零长度曲线，避免 Blender fill caps 形成结点",
    )
    args = parser.parse_args()

    strands = read_ordered_strands(args.input)
    reconstruction_strands = [strand.copy() for strand in strands]
    parting_mask = None
    front_calib = None
    selected = None
    if args.parting_mask or args.front_calib:
        if not args.parting_mask or not args.front_calib:
            raise ValueError("--parting_mask 与 --front_calib 必须同时提供")
        parting_mask = cv2.imread(args.parting_mask, cv2.IMREAD_GRAYSCALE)
        if parting_mask is None:
            raise FileNotFoundError(f"无法读取发缝 mask: {args.parting_mask}")
        parting_mask = parting_mask > 0
        front_calib = _load_projection_calibration(
            args.front_calib, image_size=1024
        )
        selected = select_roots_near_parting(
            reconstruction_strands,
            parting_mask,
            front_calib,
            radius_px=args.parting_envelope_radius_px,
        )
    if args.head_space == "blender":
        strands = _transform_strands_to_blender(strands)
    attached, report = attach_nearest_roots(
        strands,
        args.head_mesh,
        attach_steps=args.attach_steps,
        hard_steps=args.hard_steps,
        probe_steps=args.probe_steps,
        target_distance=args.target_distance,
        max_root_distance=args.max_root_distance,
        endpoint_mode=args.endpoint_mode,
    )
    if args.envelope_release_length > 0.0:
        attached, envelope_report = enforce_root_clearance_envelope(
            attached,
            args.head_mesh,
            selected=selected,
            contact_length=args.envelope_contact_length,
            release_length=args.envelope_release_length,
            contact_clearance=args.envelope_contact_clearance,
            release_clearance=args.envelope_release_clearance,
        )
        report["clearance_envelope"] = envelope_report
    if args.head_space == "blender":
        attached = _transform_strands_from_blender(attached)
    if args.enforce_parting_safe_roots:
        if parting_mask is None or front_calib is None:
            raise ValueError("发缝安全根校正需要 --parting_mask 和 --front_calib")
        attached, safe_root_report = enforce_parting_safe_roots(
            attached,
            args.head_mesh,
            parting_mask,
            front_calib,
            head_space=args.head_space,
            boundary_offset_px=args.parting_safe_boundary_offset_px,
            transition_steps=args.parting_safe_transition_steps,
            surface_distance=args.parting_safe_surface_distance,
            max_shift_distance=args.parting_safe_max_shift,
        )
        report["parting_safe_roots"] = safe_root_report
    if args.filter_parting_topology:
        if parting_mask is None or front_calib is None:
            raise ValueError("发缝拓扑过滤需要 --parting_mask 和 --front_calib")
        attached, topology_report = filter_floating_parting_segments(
            attached,
            args.head_mesh,
            parting_mask,
            front_calib,
            corridor_dilation_px=args.parting_corridor_dilation_px,
            clearance=args.parting_floating_clearance,
            protected_root_steps=args.parting_protected_root_steps,
            head_space=args.head_space,
        )
        report["parting_topology_filter"] = topology_report
    report["head_space"] = args.head_space
    report["output_strands"] = int(sum(
        (not args.drop_collapsed_strands) or _effective_last_index(strand) > 0
        for strand in attached
    ))
    write_ordered_strands(
        attached, args.output, drop_collapsed=args.drop_collapsed_strands
    )
    report_path = args.report or os.path.splitext(args.output)[0] + "_report.json"
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
