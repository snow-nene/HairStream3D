"""Axial direction sign anchoring and multi-flow ambiguity diagnostics."""
from __future__ import annotations

import numpy as np


def anchor_direction_signs(directions, anchor_directions, anchor_weights=None, min_score=0.0):
    """Choose axial signs from trusted root/tip anchors without averaging flows."""
    values = np.asarray(directions, dtype=float)
    anchors = np.asarray(anchor_directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or anchors.shape != values.shape:
        raise ValueError("directions and anchor_directions must be N-by-3")
    weights = np.ones(len(values)) if anchor_weights is None else np.asarray(anchor_weights, float).reshape(-1)
    if len(weights) != len(values) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("anchor_weights must be finite and non-negative")
    value_norm = np.linalg.norm(values, axis=1)
    anchor_norm = np.linalg.norm(anchors, axis=1)
    valid = (value_norm > 1e-8) & (anchor_norm > 1e-8) & (weights > 0)
    unit = values / np.maximum(value_norm[:, None], 1e-12)
    anchor_unit = anchors / np.maximum(anchor_norm[:, None], 1e-12)
    score = np.sum(unit * anchor_unit, axis=1)
    oriented = unit.copy()
    oriented[valid & (score < 0)] *= -1.0
    unresolved = (~valid) | (np.abs(score) < float(min_score))
    return {"directions": oriented, "sign": np.where(score < 0, -1, 1).astype(np.int8),
            "score": score, "unresolved": unresolved, "valid": valid}


def classify_multiflow_ambiguity(directions, partition_ids, *, angle_threshold_deg=35.0):
    """Flag partitions containing non-collinear axial flows; never average them."""
    values = np.asarray(directions, float)
    labels = np.asarray(partition_ids, int).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 3 or len(labels) != len(values):
        raise ValueError("directions and partition_ids must match")
    norm = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(1) & (norm > 1e-8) & (labels > 0)
    unit = values / np.maximum(norm[:, None], 1e-12)
    ambiguous = np.zeros(len(values), dtype=bool)
    candidate_count = np.ones(len(values), dtype=np.int32)
    cosine_limit = np.cos(np.deg2rad(float(angle_threshold_deg)))
    for label in np.unique(labels[valid]):
        ids = np.flatnonzero(valid & (labels == label))
        if len(ids) < 2:
            continue
        representative = unit[ids[0]]
        axial_similarity = np.abs(unit[ids] @ representative)
        bad = axial_similarity < cosine_limit
        if np.any(bad):
            ambiguous[ids] = True
            candidate_count[ids] = int(1 + np.count_nonzero(bad))
    return {"ambiguous": ambiguous, "unresolved": ~valid, "candidate_count": candidate_count,
            "valid": valid}
