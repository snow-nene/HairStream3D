"""Coverage-driven supplemental root allocation."""
from __future__ import annotations

import numpy as np


def allocate_coverage_roots(candidates, candidate_partition, candidate_visible, existing_points,
                            *, budget, min_distance=0.0, seed=42):
    """Select supplemental roots from visible coverage holes, preserving existing roots."""
    points = np.asarray(candidates, float)
    labels = np.asarray(candidate_partition, int).reshape(-1)
    visible = np.asarray(candidate_visible, bool).reshape(-1)
    existing = np.asarray(existing_points, float).reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or len(labels) != len(points) or len(visible) != len(points):
        raise ValueError("candidate arrays must match")
    if budget < 0 or min_distance < 0:
        raise ValueError("budget and min_distance must be non-negative")
    valid = np.isfinite(points).all(1) & (labels > 0) & visible
    if len(existing):
        valid &= np.min(np.linalg.norm(points[:, None] - existing[None], axis=2), axis=1) >= float(min_distance)
    pool = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    # Round-robin partitions prevents a large visible zone from consuming all budget.
    selected = []
    for label in sorted(np.unique(labels[pool]).tolist()):
        ids = pool[labels[pool] == label]
        if len(ids):
            selected.append(int(ids[0]))
    remaining = np.setdiff1d(pool, np.asarray(selected, int), assume_unique=False)
    if len(selected) < budget and len(remaining):
        remaining = rng.permutation(remaining)[: int(budget) - len(selected)]
        selected.extend(int(i) for i in remaining)
    selected = np.asarray(selected[: int(budget)], int)
    return {"indices": selected, "points": points[selected], "partitions": labels[selected],
            "candidate_count": int(len(pool)), "existing_count": int(len(existing)),
            "budget": int(budget), "strategy": "visible_coverage_round_robin"}


def coverage_gain(selected_points, target_points, radius):
    selected = np.asarray(selected_points, float).reshape(-1, 3)
    target = np.asarray(target_points, float).reshape(-1, 3)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not len(target):
        return 0.0
    if not len(selected):
        return 0.0
    covered = np.min(np.linalg.norm(target[:, None] - selected[None], axis=2), axis=1) <= radius
    return float(np.mean(covered))
