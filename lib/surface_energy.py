"""Continuous surface-energy weights for the screened-Poisson field."""
from __future__ import annotations

import numpy as np


def build_surface_normal_penalty(
    signed_distance: np.ndarray,
    normals: np.ndarray,
    *,
    support_radius: float,
    max_weight: float,
    surface_type: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return typed normal penalties with compact, distance-continuous support.

    ``signed_distance`` is metric distance from the selected surface.  Hair
    outer-surface labels receive the penalty; head/internal labels are zero so
    an internal interface cannot accidentally become an exterior barrier.
    """
    distance = np.asarray(signed_distance, dtype=np.float32)
    normal = np.asarray(normals, dtype=np.float32)
    if distance.ndim != 3 or normal.shape != (3, *distance.shape):
        raise ValueError("distance/normals shapes are incompatible")
    if support_radius <= 0 or max_weight < 0:
        raise ValueError("support_radius must be positive and max_weight non-negative")
    if not np.isfinite(distance).all() or not np.isfinite(normal).all():
        raise ValueError("distance and normals must be finite")
    labels = np.ones(distance.shape, dtype=np.int8) if surface_type is None else np.asarray(surface_type)
    if labels.shape != distance.shape:
        raise ValueError("surface_type must match signed_distance")
    magnitude = np.linalg.norm(normal, axis=0)
    unit = normal / np.maximum(magnitude[None], 1e-12)
    support = np.clip(1.0 - np.abs(distance) / float(support_radius), 0.0, 1.0)
    valid = (labels == 1) & (magnitude > 1e-8) & np.isfinite(support)
    weights = np.where(valid, float(max_weight) * support, 0.0).astype(np.float32)
    unit[:, ~valid] = 0.0
    return unit, weights
