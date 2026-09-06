"""Model-independent visible projection metrics for PDE/growth evaluation."""
from __future__ import annotations

import numpy as np


def visible_projection_metrics(points_uv, predicted_directions, target_directions,
                               visible_mask, hair_mask, *, angle_threshold_deg=30.0):
    """Compute visible axial error, coverage and leakage without changing inputs."""
    uv = np.asarray(points_uv, float).reshape(-1, 2)
    pred = np.asarray(predicted_directions, float).reshape(-1, 3)
    target = np.asarray(target_directions, float).reshape(-1, 3)
    visible = np.asarray(visible_mask, bool).reshape(-1)
    hair = np.asarray(hair_mask, bool)
    if len(uv) != len(pred) or pred.shape != target.shape or len(visible) != len(uv):
        raise ValueError("projection arrays must have matching lengths")
    valid = np.isfinite(uv).all(1) & np.isfinite(pred).all(1) & np.isfinite(target).all(1)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
    if hair.ndim != 2:
        raise ValueError("hair_mask must be a 2-D image mask")
    inside &= (uv[:, 0] < hair.shape[1]) & (uv[:, 1] < hair.shape[0])
    ids = np.flatnonzero(inside & visible)
    pn = np.linalg.norm(pred[ids], axis=1)
    tn = np.linalg.norm(target[ids], axis=1)
    usable = (pn > 1e-8) & (tn > 1e-8)
    dots = np.abs(np.sum(pred[ids] * target[ids], axis=1) / np.maximum(pn * tn, 1e-12))
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    px = np.rint(uv[inside, 0]).astype(int)
    py = np.rint(uv[inside, 1]).astype(int)
    in_hair = hair[py, px]
    leakage = int(np.count_nonzero(~in_hair))
    return {
        "projected": int(len(uv)), "valid_projection": int(np.count_nonzero(inside)),
        "visible_samples": int(len(ids)), "usable_direction_samples": int(np.count_nonzero(usable)),
        "coverage": float(len(ids) / max(1, np.count_nonzero(visible))),
        "leakage_rate": float(leakage / max(1, np.count_nonzero(inside))),
        "out_of_bounds": int(np.count_nonzero(valid & ~inside)),
        "axial_angle_median_deg": float(np.median(angles[usable])) if np.any(usable) else None,
        "axial_angle_q95_deg": float(np.quantile(angles[usable], .95)) if np.any(usable) else None,
        "axial_within_threshold": float(np.mean(angles[usable] <= angle_threshold_deg)) if np.any(usable) else 0.0,
    }
