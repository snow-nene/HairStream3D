"""Local, opt-in groom layer for keeping a reconstructed parting narrow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
import trimesh


@dataclass(frozen=True)
class GroomConfig:
    strands_per_side: int = 128
    root_bank_min_m: float = 0.004
    root_bank_max_m: float = 0.012
    guide_root_max_m: float = 0.035
    dense_length_m: float = 0.020
    root_spacing_m: float = 0.001
    side_lock_length_m: float = 0.012
    opening_control_m: float = 0.030
    handoff_m: float = 0.060
    max_added_opening_m: float = 0.002
    minimum_side_distance_m: float = 0.00025
    surface_root_clearance_m: float = 0.0005
    surface_peak_clearance_m: float = 0.004
    curve_margin_fraction: float = 0.05
    coverage_bins: int = 16


def read_ordered_strands(path: str | Path) -> list[np.ndarray]:
    """Recover ordered polylines from an Open3D line-set PLY."""

    line_set = o3d.io.read_line_set(str(path))
    points = np.asarray(line_set.points, dtype=np.float64)
    lines = np.asarray(line_set.lines, dtype=np.int64)
    if len(points) == 0 or len(lines) == 0:
        raise ValueError(f"PLY 不包含有效折线: {path}")

    strands: list[np.ndarray] = []
    current = [int(lines[0, 0]), int(lines[0, 1])]
    for previous, edge in zip(lines[:-1], lines[1:]):
        if int(edge[0]) == int(previous[1]):
            current.append(int(edge[1]))
        else:
            strands.append(points[np.asarray(current, dtype=np.int64)].copy())
            current = [int(edge[0]), int(edge[1])]
    strands.append(points[np.asarray(current, dtype=np.int64)].copy())
    return strands


def write_ordered_strands(strands: Iterable[np.ndarray], path: str | Path) -> None:
    """Write variable-length ordered polylines without changing their points."""

    point_parts: list[np.ndarray] = []
    lines: list[list[int]] = []
    offset = 0
    for strand in strands:
        strand = np.asarray(strand, dtype=np.float64)
        if strand.ndim != 2 or strand.shape[1] != 3 or len(strand) < 2:
            raise ValueError("每根发束必须是形状为 (N, 3) 且 N >= 2 的数组")
        point_parts.append(strand)
        lines.extend(
            [offset + index, offset + index + 1]
            for index in range(len(strand) - 1)
        )
        offset += len(strand)
    if not point_parts:
        raise ValueError("不能写出空发束集合")

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.concatenate(point_parts, axis=0)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_line_set(str(output), line_set, write_ascii=False):
        raise RuntimeError(f"无法写出 PLY: {output}")


def load_topology_roots(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"points", "side", "parting_curve"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"root_prefixes.npz 缺少字段: {missing}")
    points = np.asarray(data["points"], dtype=np.float64)
    side = np.asarray(data["side"], dtype=np.int8)
    curve = np.asarray(data["parting_curve"], dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3 or len(points) != len(side):
        raise ValueError("拓扑根点数组形状无效")
    return {"points": points, "side": side, "parting_curve": curve}


def _normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def _effective_strand(strand: np.ndarray) -> np.ndarray:
    strand = np.asarray(strand, dtype=np.float64)
    moving = np.flatnonzero(np.linalg.norm(np.diff(strand, axis=0), axis=1) > 1e-8)
    last = int(moving[-1] + 1) if len(moving) else 0
    if last < 1:
        raise ValueError("导向发束没有有效长度")
    return strand[: last + 1].copy()


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def build_curve_lateral_frame(
    curve: np.ndarray,
    roots: np.ndarray,
    side: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """Estimate the +bank lateral axis from labelled topology roots."""

    curve = np.asarray(curve, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    nearest = np.argmin(
        np.sum((roots[:, None, :] - curve[None, :, :]) ** 2, axis=2), axis=1
    )
    directions = _normalize(roots - curve[nearest]) * side[:, None]
    lateral = np.zeros_like(curve)
    for curve_id in range(len(curve)):
        local = np.abs(nearest - curve_id) <= int(window)
        if np.any(local):
            lateral[curve_id] = np.mean(directions[local], axis=0)
        elif curve_id:
            lateral[curve_id] = lateral[curve_id - 1]
        else:
            lateral[curve_id] = directions[0]
    for _ in range(3):
        padded = np.pad(lateral, ((1, 1), (0, 0)), mode="edge")
        lateral = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
        lateral = _normalize(lateral)
    return lateral


def curve_coordinates(
    points: np.ndarray, curve: np.ndarray, lateral: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    nearest = np.argmin(
        np.sum((flat[:, None, :] - curve[None, :, :]) ** 2, axis=2), axis=1
    )
    delta = flat - curve[nearest]
    signed = np.sum(delta * lateral[nearest], axis=1)
    distance = np.linalg.norm(delta, axis=1)
    shape = points.shape[:-1]
    return signed.reshape(shape), distance.reshape(shape), nearest.reshape(shape)


def select_balanced_roots(
    roots: np.ndarray,
    side: np.ndarray,
    curve: np.ndarray,
    lateral: np.ndarray,
    config: GroomConfig,
) -> np.ndarray:
    """Select equal, approximately curve-uniform candidates from both banks."""

    signed, _, nearest = curve_coordinates(roots, curve, lateral)
    margin = int(round((len(curve) - 1) * config.curve_margin_fraction))
    selected: list[int] = []
    target_positions = np.linspace(
        margin, len(curve) - 1 - margin, config.strands_per_side
    )
    for bank_side in (-1, 1):
        eligible = np.flatnonzero(
            (side == bank_side)
            & (np.abs(signed) >= config.root_bank_min_m)
            & (np.abs(signed) <= config.root_bank_max_m)
            & (nearest >= margin)
            & (nearest <= len(curve) - 1 - margin)
        )
        if len(eligible) < config.strands_per_side:
            raise ValueError(
                f"侧别 {bank_side} 只有 {len(eligible)} 个合格根点，"
                f"少于请求的 {config.strands_per_side}"
            )
        unused = set(int(index) for index in eligible)
        bank_mid = 0.5 * (config.root_bank_min_m + config.root_bank_max_m)
        for target in target_positions:
            choices = np.fromiter(unused, dtype=np.int64)
            score = np.abs(nearest[choices] - target)
            score += 0.05 * np.abs(np.abs(signed[choices]) - bank_mid) / max(
                config.root_bank_max_m - config.root_bank_min_m, 1e-9
            )
            winner = int(choices[np.argmin(score)])
            selected.append(winner)
            unused.remove(winner)
    return np.asarray(selected, dtype=np.int64)


def match_guides(
    selected_roots: np.ndarray,
    selected_side: np.ndarray,
    guides: list[np.ndarray],
    curve: np.ndarray,
    lateral: np.ndarray,
    config: GroomConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically match every new root to one unique same-bank guide."""

    guide_roots = np.stack([np.asarray(strand)[0] for strand in guides])
    guide_signed, guide_curve_distance, _ = curve_coordinates(
        guide_roots, curve, lateral
    )
    guide_side = np.where(guide_signed >= 0.0, 1, -1)
    guide_length = np.zeros(len(guides), dtype=np.float64)
    for index, guide in enumerate(guides):
        try:
            guide_length[index] = _arc_lengths(_effective_strand(guide))[-1]
        except ValueError:
            guide_length[index] = 0.0
    valid = (
        (guide_curve_distance <= config.guide_root_max_m)
        & (guide_length > config.handoff_m + config.root_spacing_m)
    )
    used: set[int] = set()
    matched: list[int] = []
    distances: list[float] = []
    for root, bank_side in zip(selected_roots, selected_side):
        candidates = np.flatnonzero(valid & (guide_side == int(bank_side)))
        candidates = np.asarray(
            [index for index in candidates if int(index) not in used],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            raise ValueError(f"侧别 {int(bank_side)} 没有足够的唯一 v30 导向发束")
        delta = guide_roots[candidates] - root
        candidate_distance = np.linalg.norm(delta, axis=1)
        winner_pos = int(np.argmin(candidate_distance))
        winner = int(candidates[winner_pos])
        matched.append(winner)
        distances.append(float(candidate_distance[winner_pos]))
        used.add(winner)
    return np.asarray(matched, dtype=np.int64), np.asarray(distances)


def _densify_guide(
    guide: np.ndarray, dense_length_m: float, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    guide = _effective_strand(guide)
    arc = _arc_lengths(guide)
    dense_stop = min(float(dense_length_m), float(arc[-1]))
    dense_arc = np.arange(0.0, dense_stop, float(spacing_m))
    dense_arc = np.unique(np.concatenate([dense_arc, [dense_stop]]))
    dense_points = np.column_stack(
        [np.interp(dense_arc, arc, guide[:, axis]) for axis in range(3)]
    )
    tail_ids = np.flatnonzero(arc > dense_stop + 1e-10)
    if len(tail_ids):
        points = np.concatenate([dense_points, guide[tail_ids]], axis=0)
        point_arc = np.concatenate([dense_arc, arc[tail_ids]])
    else:
        points, point_arc = dense_points, dense_arc
    return points, point_arc


def groom_guide(
    guide: np.ndarray,
    target_root: np.ndarray,
    bank_side: int,
    curve: np.ndarray,
    lateral: np.ndarray,
    head_mesh: trimesh.Trimesh,
    config: GroomConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Relocate and narrow one guide root with an exact unmodified tail."""

    base, arc = _densify_guide(
        guide,
        max(config.dense_length_m, config.handoff_m),
        config.root_spacing_m,
    )
    if arc[-1] <= config.handoff_m:
        raise ValueError("导向发束短于配置的交接弧长")

    release = 1.0 - _smoothstep01(arc / config.handoff_m)
    moved = base + release[:, None] * (target_root - base[0])[None, :]
    root_signed, _, _ = curve_coordinates(
        np.asarray(target_root)[None, :], curve, lateral
    )
    root_bank = max(abs(float(root_signed[0])), config.minimum_side_distance_m)
    limit = root_bank + config.max_added_opening_m

    # Root relocation and surface clearance both change physical arc length.
    # Alternate the side-cap and outside-surface projections so neither one
    # invalidates the other on the final curve.
    groomed = moved.copy()
    for _ in range(8):
        output_arc = _arc_lengths(groomed)
        signed, _, nearest = curve_coordinates(groomed, curve, lateral)
        magnitude = int(bank_side) * signed
        desired_magnitude = np.minimum(magnitude, limit)
        locked = output_arc <= config.side_lock_length_m
        desired_magnitude[locked] = np.clip(
            magnitude[locked], config.minimum_side_distance_m, limit
        )
        desired_signed = int(bank_side) * desired_magnitude
        opening_release = np.ones_like(output_arc)
        transition = output_arc > config.opening_control_m
        opening_release[transition] = 1.0 - _smoothstep01(
            (output_arc[transition] - config.opening_control_m)
            / max(config.handoff_m - config.opening_control_m, 1e-9)
        )
        opening_release[arc >= config.handoff_m] = 0.0
        groomed = groomed + (
            opening_release * (desired_signed - signed)
        )[:, None] * lateral[nearest]

        # Keep the modified segment outside the head. A smooth clearance hump
        # makes the root layer renderable and returns to root clearance at handoff.
        output_arc = _arc_lengths(groomed)
        surface_ids = np.flatnonzero(
            (output_arc < config.handoff_m) & (arc < config.handoff_m)
        )
        closest, _, face_ids = trimesh.proximity.closest_point(
            head_mesh, groomed[surface_ids]
        )
        normals = np.asarray(head_mesh.face_normals, dtype=np.float64)[face_ids]
        signed_height = np.sum(
            (groomed[surface_ids] - closest) * normals, axis=1
        )
        rise = _smoothstep01(output_arc[surface_ids] / config.opening_control_m)
        fall = 1.0 - _smoothstep01(
            (output_arc[surface_ids] - config.opening_control_m)
            / max(config.handoff_m - config.opening_control_m, 1e-9)
        )
        hump = np.where(
            output_arc[surface_ids] <= config.opening_control_m, rise, fall
        )
        target_height = config.surface_root_clearance_m + hump * (
            config.surface_peak_clearance_m - config.surface_root_clearance_m
        )
        lift = np.maximum(target_height - signed_height, 0.0)
        groomed[surface_ids] += lift[:, None] * normals
        groomed[0] = target_root

    handoff_id = int(np.searchsorted(arc, config.handoff_m, side="left"))
    tail = arc >= config.handoff_m
    groomed[tail] = base[tail]
    before = _normalize((groomed[handoff_id] - groomed[handoff_id - 1])[None, :])[0]
    guide_tangent = _normalize((base[handoff_id] - base[handoff_id - 1])[None, :])[0]
    angle = float(
        np.degrees(np.arccos(np.clip(np.dot(before, guide_tangent), -1.0, 1.0)))
    )
    diagnostics: dict[str, float | int] = {
        "handoff_index": handoff_id,
        "handoff_angle_deg": angle,
        "tail_max_deviation_m": float(
            np.max(np.linalg.norm(groomed[tail] - base[tail], axis=1))
        ),
    }
    return groomed, diagnostics


def _sample_at_arc(strand: np.ndarray, target: float) -> np.ndarray:
    arc = _arc_lengths(strand)
    clipped = min(float(target), float(arc[-1]))
    return np.asarray(
        [np.interp(clipped, arc, strand[:, axis]) for axis in range(3)]
    )


def _profile_metrics(
    strands: list[np.ndarray],
    side: np.ndarray,
    curve: np.ndarray,
    lateral: np.ndarray,
    distances_m: tuple[float, ...],
    coverage_bins: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    minimum_bank_count = min(int(np.sum(side == bank)) for bank in (-1, 1))
    effective_bins = max(1, min(int(coverage_bins), minimum_bank_count // 2))
    for target in distances_m:
        points = np.stack([_sample_at_arc(strand, target) for strand in strands])
        signed, _, nearest = curve_coordinates(points, curve, lateral)
        edges = np.linspace(
            float(np.min(nearest)), float(np.max(nearest)) + 1.0, effective_bins + 1
        )
        bank_medians = {
            int(bank): float(np.median(np.abs(signed[side == bank])))
            for bank in (-1, 1)
        }
        covered = []
        for begin, end in zip(edges[:-1], edges[1:]):
            in_bin = (nearest >= begin) & (nearest < end)
            covered.append(
                bool(np.any(in_bin & (side == -1)) and np.any(in_bin & (side == 1)))
            )
        result[f"{int(round(target * 1000.0))}mm"] = {
            "gap_m": bank_medians[-1] + bank_medians[1],
            "left_bank_median_m": bank_medians[-1],
            "right_bank_median_m": bank_medians[1],
            "both_bank_coverage": float(np.mean(covered)),
        }
    return result


def evaluate_layer(
    strands: list[np.ndarray],
    side: np.ndarray,
    curve: np.ndarray,
    lateral: np.ndarray,
    head_mesh: trimesh.Trimesh,
    match_distances: np.ndarray,
    diagnostics: list[dict[str, float | int]],
    config: GroomConfig,
) -> dict[str, object]:
    roots = np.stack([strand[0] for strand in strands])
    _, surface_distance, _ = trimesh.proximity.closest_point(head_mesh, roots)
    violations = 0
    locked_points = 0
    surface_heights: list[np.ndarray] = []
    for strand, bank_side in zip(strands, side):
        arc = _arc_lengths(strand)
        locked = arc <= config.side_lock_length_m
        signed, _, _ = curve_coordinates(strand[locked], curve, lateral)
        violations += int(np.sum(int(bank_side) * signed < -1e-8))
        locked_points += int(np.sum(locked))
        surface_layer = arc < config.handoff_m
        closest, _, face_ids = trimesh.proximity.closest_point(
            head_mesh, strand[surface_layer]
        )
        normals = np.asarray(head_mesh.face_normals, dtype=np.float64)[face_ids]
        surface_heights.append(
            np.sum((strand[surface_layer] - closest) * normals, axis=1)
        )

    surface_height = np.concatenate(surface_heights)

    profile = _profile_metrics(
        strands,
        side,
        curve,
        lateral,
        (0.0, 0.006, 0.012, 0.018, 0.030, 0.048),
        config.coverage_bins,
    )
    root_gap = float(profile["0mm"]["gap_m"])
    opening_ratio = float(profile["30mm"]["gap_m"]) / max(root_gap, 1e-12)
    counts = {str(bank): int(np.sum(side == bank)) for bank in (-1, 1)}
    failures: list[str] = []
    if violations:
        failures.append("root_side_violations")
    if int(np.sum(surface_height < -1e-5)):
        failures.append("root_segment_surface_penetration")
    if float(np.median(surface_distance)) > 0.001:
        failures.append("root_surface_distance_q50")
    if opening_ratio > 1.5:
        failures.append("opening_ratio_30mm")
    if float(profile["0mm"]["both_bank_coverage"]) < 0.95:
        failures.append("both_bank_root_coverage")
    if max(float(item["handoff_angle_deg"]) for item in diagnostics) > 20.0:
        failures.append("handoff_angle")
    if max(float(item["tail_max_deviation_m"]) for item in diagnostics) > 1e-9:
        failures.append("tail_deviation")

    return {
        "passed": not failures,
        "failed_gates": failures,
        "strand_count": int(len(strands)),
        "side_counts": counts,
        "match_distance_m": {
            "q50": float(np.quantile(match_distances, 0.50)),
            "q95": float(np.quantile(match_distances, 0.95)),
            "max": float(np.max(match_distances)),
        },
        "root_surface_distance_m": {
            "q50": float(np.quantile(surface_distance, 0.50)),
            "q95": float(np.quantile(surface_distance, 0.95)),
            "max": float(np.max(surface_distance)),
        },
        "root_layer": {
            "side_violations": int(violations),
            "locked_points": int(locked_points),
            "side_violation_rate": float(violations / max(locked_points, 1)),
            "surface_penetration_points": int(np.sum(surface_height < -1e-5)),
            "surface_signed_height_q01_m": float(
                np.quantile(surface_height, 0.01)
            ),
            "surface_signed_height_q50_m": float(
                np.quantile(surface_height, 0.50)
            ),
        },
        "gap_profile": profile,
        "opening_ratio_30mm": opening_ratio,
        "handoff_angle_deg": {
            "q50": float(np.median([item["handoff_angle_deg"] for item in diagnostics])),
            "q95": float(np.quantile([item["handoff_angle_deg"] for item in diagnostics], 0.95)),
            "max": float(max(item["handoff_angle_deg"] for item in diagnostics)),
        },
        "tail_max_deviation_m": float(
            max(item["tail_max_deviation_m"] for item in diagnostics)
        ),
    }


def build_local_parting_layer(
    guides: list[np.ndarray],
    topology_points: np.ndarray,
    topology_side: np.ndarray,
    curve: np.ndarray,
    head_mesh: trimesh.Trimesh,
    config: GroomConfig | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    config = config or GroomConfig()
    roots = np.asarray(topology_points, dtype=np.float64)[:, 0]
    lateral = build_curve_lateral_frame(curve, roots, topology_side)
    selected_ids = select_balanced_roots(
        roots, topology_side, curve, lateral, config
    )
    selected_roots = roots[selected_ids]
    selected_side = np.asarray(topology_side[selected_ids], dtype=np.int8)
    guide_ids, match_distances = match_guides(
        selected_roots, selected_side, guides, curve, lateral, config
    )

    layer: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int]] = []
    for root, bank_side, guide_id in zip(
        selected_roots, selected_side, guide_ids
    ):
        strand, strand_diagnostics = groom_guide(
            guides[int(guide_id)],
            root,
            int(bank_side),
            curve,
            lateral,
            head_mesh,
            config,
        )
        layer.append(strand)
        diagnostics.append(strand_diagnostics)

    valid_handoff = np.asarray(
        [float(item["handoff_angle_deg"]) <= 20.0 for item in diagnostics]
    )
    valid_per_side = {
        bank: np.flatnonzero(valid_handoff & (selected_side == bank))
        for bank in (-1, 1)
    }
    balanced_count = min(len(indices) for indices in valid_per_side.values())
    if balanced_count == 0:
        raise ValueError("交接角筛选后至少一侧没有可用发束")
    keep_parts = []
    for bank in (-1, 1):
        candidates = valid_per_side[bank]
        angle = np.asarray(
            [float(diagnostics[index]["handoff_angle_deg"]) for index in candidates]
        )
        keep_parts.append(candidates[np.argsort(angle)[:balanced_count]])
    keep = np.sort(np.concatenate(keep_parts))
    rejected_handoff = int(len(layer) - len(keep))
    layer = [layer[index] for index in keep]
    diagnostics = [diagnostics[index] for index in keep]
    selected_ids = selected_ids[keep]
    selected_side = selected_side[keep]
    guide_ids = guide_ids[keep]
    match_distances = match_distances[keep]

    metrics = evaluate_layer(
        layer,
        selected_side,
        curve,
        lateral,
        head_mesh,
        match_distances,
        diagnostics,
        config,
    )
    metrics["selected_topology_ids"] = selected_ids.tolist()
    metrics["matched_guide_ids"] = guide_ids.tolist()
    metrics["handoff_rejected_strands"] = rejected_handoff
    return layer, metrics
