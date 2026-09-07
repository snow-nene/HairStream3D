"""Coverage-driven supplemental growth planning with explicit stop reasons."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lib.coverage_budget import allocate_coverage_roots, coverage_gain
from lib.guide_quality import filter_guides


def plan_supplemental_growth(candidates, partitions, visible, existing_roots, target_points,
                             guides, guide_root_labels, guide_sample_labels, guide_confidence,
                             *, budget, min_confidence=.5, radius=.01, seed=42):
    guide_report = filter_guides(guides, guide_root_labels, guide_sample_labels, guide_confidence,
                                 min_confidence=min_confidence)
    supported_labels = {item["root_partition"] for item in guide_report["accepted"]}
    eligible = np.asarray(visible, bool) & np.isin(partitions, list(supported_labels))
    targets = np.asarray(target_points, float).reshape(-1, 3)
    points = np.asarray(candidates, float).reshape(-1, 3)
    if not np.isfinite(targets).all() or radius <= 0:
        raise ValueError("finite coverage targets and positive radius required")
    near_target = np.zeros(len(points), bool)
    finite = np.isfinite(points).all(axis=1)
    if len(targets):
        near_target[finite] = cKDTree(targets).query(points[finite])[0] <= radius
    eligible &= near_target
    roots = allocate_coverage_roots(candidates, partitions, eligible, existing_roots,
                                    budget=budget, min_distance=radius, seed=seed)
    gain = coverage_gain(roots["points"], target_points, radius)
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(eligible)
    random_ids = rng.permutation(pool)[:len(roots["indices"])] if len(pool) else np.empty(0, int)
    random_gain = coverage_gain(np.asarray(candidates)[random_ids], target_points, radius)
    stop_reason = "budget_exhausted" if len(roots["indices"]) >= budget else "no_visible_partition_candidates"
    if not len(guide_report["accepted_ids"]):
        stop_reason = "no_usable_guides"
    return {"roots": roots, "guides": guide_report, "coverage_gain": gain,
            "random_baseline_gain": random_gain, "stop_reason": stop_reason,
            "existing_root_count": int(len(existing_roots)), "seed": int(seed)}


def execute_supplemental_growth(plan, *, query_field, check_segment,
                                audit_trajectory, coverage_score, existing_strands,
                                step_m=.0005, length_m=.1, min_length_m=.01,
                                coverage_delta=None, coverage_commit=None,
                                trajectory_quality=None):
    """Integrate new roots without copying or attracting to guide positions.

    query_field(point, partition) returns a directed vector and a stop reason
    (None when valid). check_segment(start, end, partition) checks the entire
    swept segment, including head clearance/domain/partition, and returns a
    reason or None. audit_trajectory is an independent export gate. The
    coverage callback measures visible coverage for the complete strand set;
    only audited trajectories with positive marginal coverage are accepted.
    These callbacks must share world coordinates and the actual template.
    """
    if not (np.isfinite([step_m, length_m, min_length_m]).all()
            and step_m > 0 and length_m >= min_length_m > 0):
        raise ValueError("invalid integration lengths")
    existing = [np.asarray(s, float).copy() for s in existing_strands]
    accepted, records = [], []
    baseline = float(coverage_score(existing))
    if not np.isfinite(baseline):
        raise ValueError("coverage score must be finite")
    initial_score = baseline
    for root_id, root, partition in zip(plan["roots"]["indices"],
                                        plan["roots"]["points"],
                                        plan["roots"]["partitions"]):
        point = np.asarray(root, float).copy()
        path = [point.copy()]
        reason = check_segment(point, point, int(partition))
        length = 0.
        while reason is None and length < length_m - 1e-12:
            direction, reason = query_field(point, int(partition))
            direction = np.asarray(direction, float)
            if reason is not None:
                break
            norm = np.linalg.norm(direction)
            if direction.shape != (3,) or not np.isfinite(direction).all() or norm < 1e-10:
                reason = "zero_or_invalid_direction"
                break
            step = min(step_m, length_m - length)
            midpoint = point + .5 * step * direction / norm
            reason = check_segment(point, midpoint, int(partition))
            if reason is not None:
                break
            tangent, reason = query_field(midpoint, int(partition))
            if reason is not None:
                break
            tangent = np.asarray(tangent, float)
            norm = np.linalg.norm(tangent)
            if tangent.shape != (3,) or not np.isfinite(tangent).all() or norm < 1e-10:
                reason = "zero_or_invalid_direction"
                break
            endpoint = point + step * tangent / norm
            reason = check_segment(point, endpoint, int(partition))
            if reason is not None:
                break
            path.append(endpoint.copy())
            point = endpoint
            length += step
        trajectory = np.asarray(path)
        rejection = None
        audit = None
        gain = 0.
        if length < min_length_m:
            rejection = "too_short"
        elif trajectory_quality is not None and not trajectory_quality(existing + accepted, trajectory):
            rejection = "trajectory_quality"
        else:
            audit = audit_trajectory(trajectory)
            if not isinstance(audit, dict) or audit.get("passed") is not True:
                rejection = "independent_collision_audit"
            else:
                if coverage_delta is None:
                    score = float(coverage_score(existing + accepted + [trajectory]))
                else:
                    score = baseline + float(coverage_delta(existing + accepted, trajectory))
                if not np.isfinite(score) or score <= baseline:
                    rejection = "no_visible_coverage_gain"
                else:
                    gain = score - baseline
                    baseline = score
                    accepted.append(trajectory)
                    if coverage_commit is not None:
                        coverage_commit(trajectory)
        records.append({"candidate_id": int(root_id), "partition": int(partition),
                        "length_m": length, "stop_reason": reason or "length_budget",
                        "rejection": rejection, "accepted": rejection is None,
                        "coverage_gain": gain, "collision_audit": audit})
    return {"strands": accepted, "records": records,
            "coverage_before": initial_score, "coverage_after": baseline,
            "existing_root_count": len(existing), "accepted_count": len(accepted)}
