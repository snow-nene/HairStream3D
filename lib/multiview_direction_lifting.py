"""Training-free 2-D-to-3-D direction lifting on visible mesh intersections."""
from __future__ import annotations

import numpy as np


def lift_visible_directions(pixel_y, pixel_x, world_points, partition_ids,
                            strand_rgb, image_shape, step_px=2,
                            min_length_m=1e-8):
    """Lift HairStep image directions using neighboring visible intersections.

    Pixel coordinates are integer ``(y, x)``. A neighbor is accepted only when
    it is present in the visible-intersection cache and has the same local
    partition. The result is axial (sign-invariant) until a later root/guide
    stage chooses a directed sign.
    """
    y = np.asarray(pixel_y, dtype=int).reshape(-1)
    x = np.asarray(pixel_x, dtype=int).reshape(-1)
    world = np.asarray(world_points, dtype=float)
    labels = np.asarray(partition_ids, dtype=int).reshape(-1)
    rgb = np.asarray(strand_rgb)
    height, width = image_shape
    if (world.shape != (len(y), 3) or labels.shape != (len(y),)
            or rgb.shape != (height, width, 3) or len(y) == 0):
        raise ValueError("visible direction inputs have incompatible shapes")
    if step_px < 1 or min_length_m <= 0 or not np.isfinite(world).all():
        raise ValueError("invalid lifting parameters or world coordinates")
    if np.any((y < 0) | (y >= height) | (x < 0) | (x >= width)):
        raise ValueError("pixel coordinates outside image")
    # HairStep encoding used by the repository: dx=1-2B, dy=2G-1.
    unit = rgb.astype(float) / 255.0
    dx = 1.0 - 2.0 * unit[..., 2]
    dy = 2.0 * unit[..., 1] - 1.0
    norm = np.hypot(dx, dy)
    lookup = {(int(yy), int(xx)): i for i, (yy, xx) in enumerate(zip(y, x))}
    lifted = np.zeros_like(world, dtype=float)
    accepted = np.zeros(len(y), dtype=bool)
    reason = np.full(len(y), "invalid_direction", dtype=object)
    for i, (yy, xx) in enumerate(zip(y, x)):
        if not np.isfinite(norm[yy, xx]) or norm[yy, xx] <= 1e-6:
            continue
        nx = int(round(xx + step_px * dx[yy, xx] / norm[yy, xx]))
        ny = int(round(yy + step_px * dy[yy, xx] / norm[yy, xx]))
        if not (0 <= nx < width and 0 <= ny < height):
            reason[i] = "neighbor_outside_image"
            continue
        j = lookup.get((ny, nx))
        if j is None:
            reason[i] = "neighbor_not_visible"
            continue
        if labels[j] <= 0 or labels[i] != labels[j]:
            reason[i] = "partition_or_layer_jump"
            continue
        delta = world[j] - world[i]
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length < min_length_m:
            reason[i] = "zero_or_invalid_3d_difference"
            continue
        lifted[i] = delta / length
        accepted[i] = True
        reason[i] = "accepted"
    return {"directions": lifted.astype(np.float32), "accepted": accepted,
            "reason": reason, "step_px": int(step_px),
            "axis_order": "world_xyz"}


def compare_axial_directions(first, second, accepted=None, max_angle_deg=30.0):
    """Compare two 3-D direction estimates without treating +/- as different."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("direction arrays must both be N-by-3")
    if not 0 < max_angle_deg <= 90:
        raise ValueError("max_angle_deg must be in (0, 90]")
    usable = np.isfinite(first).all(1) & np.isfinite(second).all(1)
    usable &= np.linalg.norm(first, axis=1) > 1e-8
    usable &= np.linalg.norm(second, axis=1) > 1e-8
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool)
        if accepted.shape != (len(first),):
            raise ValueError("accepted mask shape mismatch")
        usable &= accepted
    a = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    b = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0, 1)))
    compatible = usable & (angle <= max_angle_deg)
    return {"angle_deg": angle.astype(np.float32), "usable": usable,
            "compatible": compatible, "max_angle_deg": float(max_angle_deg),
            "compatible_fraction": float(compatible.sum() / max(1, usable.sum()))}
