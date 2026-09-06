"""Optional human curve constraints with visibility and partition gates."""
from __future__ import annotations

import numpy as np


def validate_manual_curves(curves, projected_pixels, visible_mask, partition_labels, *, view_id="front"):
    """Accept only curve segments fully visible and within one partition."""
    visibility = np.asarray(visible_mask, bool)
    partitions = np.asarray(partition_labels, int)
    accepted, rejected = [], []
    for curve_id, (curve, pixels) in enumerate(zip(curves, projected_pixels)):
        points = np.asarray(curve, float)
        uv = np.asarray(pixels, int).reshape(-1, 2)
        reason = None
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(uv) or len(points) < 2:
            reason = "invalid_curve_shape"
        elif np.any(uv[:, 0] < 0) or np.any(uv[:, 1] < 0) or np.any(uv[:, 1] >= visibility.shape[0]) or np.any(uv[:, 0] >= visibility.shape[1]):
            reason = "out_of_bounds"
        elif not np.all(visibility[uv[:, 1], uv[:, 0]]):
            reason = "occluded"
        else:
            labels = partitions[uv[:, 1], uv[:, 0]]
            if np.any(labels <= 0):
                reason = "unassigned_partition"
            elif len(np.unique(labels)) != 1:
                reason = "cross_partition"
        record = {"curve_id": int(curve_id), "view": str(view_id), "reason": reason}
        (rejected if reason else accepted).append(record)
    return {"accepted": accepted, "rejected": rejected, "automatic_path_unchanged": True}


def affected_region_mask(curves, grid_shape, world_to_grid, radius_voxels=2):
    """Build a conservative local re-solve mask around accepted world curves."""
    if radius_voxels < 0:
        raise ValueError("radius_voxels must be non-negative")
    shape = tuple(int(x) for x in grid_shape)
    transform = np.asarray(world_to_grid, float)
    if len(shape) != 3 or transform.shape != (4, 4):
        raise ValueError("invalid grid shape or transform")
    mask = np.zeros(shape, bool)
    for curve in curves:
        points = np.asarray(curve, float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("curves must contain N-by-3 points")
        homogeneous = np.c_[points, np.ones(len(points))] @ transform.T
        for point in homogeneous[:, :3]:
            center = np.rint(point).astype(int)
            lower = np.maximum(center - int(radius_voxels), 0)
            upper = np.minimum(center + int(radius_voxels) + 1, np.asarray(shape))
            if np.all(lower < upper):
                mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = True
    return mask
