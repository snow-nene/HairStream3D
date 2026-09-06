"""Typed array contract separating geometry, direction, semantic and partition data."""
from __future__ import annotations

import numpy as np


def validate_volume_interfaces(geometry, direction, semantic, partition):
    """Validate independent volume channels before PDE or integration."""
    geom = np.asarray(geometry)
    field = np.asarray(direction)
    sem = np.asarray(semantic)
    labels = np.asarray(partition)
    if geom.ndim != 3 or field.shape != (3, *geom.shape):
        raise ValueError("geometry/direction shapes are incompatible")
    if sem.shape != geom.shape or labels.shape != geom.shape:
        raise ValueError("semantic and partition shapes must match geometry")
    if not np.isfinite(geom).all() or not np.isfinite(field).all():
        raise ValueError("geometry and direction must be finite")
    if np.any(labels[sem == 1] <= 0):
        raise ValueError("hair voxels require positive partition ids")
    # semantic: 0 unknown, 1 hair, 2 head/entity, 3 reliable air
    if np.any(~np.isin(sem, (0, 1, 2, 3))):
        raise ValueError("semantic labels must be 0..3")
    entity = sem == 2
    return {
        "geometry": geom,
        "direction": np.where(entity[None], 0.0, field),
        "semantic": sem.astype(np.int8, copy=False),
        "partition": labels.astype(np.int32, copy=False),
        "entity_direction_suppressed": int(entity.sum()),
    }
