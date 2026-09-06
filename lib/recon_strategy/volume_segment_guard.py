"""Conservative nearest-node voxel traversal shared by tracing and export.

Checks all cells touched at face/edge/corner events, including stationary
segments on a grid plane. Bounding box limits are node centres, not padded cells.
"""
from itertools import product

import numpy as np
import torch


def first_invalid_segment_fraction(origins, ends, root_labels, labels, low, high):
    """Batched DDA returning first illegal t in [0,1], or infinity.

    Tensor inputs are N-by-3; the label volume is XYZ, with zero outside domain.
    A small grid-space tolerance conservatively treats almost-tied events alike.
    No endpoint-only or monotonic-validity assumption is used.
    """
    device = origins.device
    a = origins.to(torch.float64)
    b = ends.to(torch.float64)
    low = torch.as_tensor(low, device=device, dtype=torch.float64).reshape(3)
    high = torch.as_tensor(high, device=device, dtype=torch.float64).reshape(3)
    shape = torch.tensor(labels.shape, device=device, dtype=torch.long)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape or labels.ndim != 3:
        raise ValueError("Expected N-by-3 segments and XYZ labels")
    if torch.any(high <= low) or torch.any(shape < 2):
        raise ValueError("Invalid volume bounds")
    roots = root_labels.to(device=device).reshape(-1)
    if roots.numel() != len(a) or torch.any(roots <= 0):
        raise ValueError("Expected positive root labels")
    spacing = (high - low) / (shape - 1)
    start = (a - low) / spacing
    delta = (b - a) / spacing
    step = torch.sign(delta).long()
    tol = 1e-7
    answer = torch.full((len(a),), float("inf"), device=device, dtype=torch.float64)
    valid_start = (torch.isfinite(a).all(1) & torch.isfinite(b).all(1)
                   & (a >= low).all(1) & (a <= high).all(1))
    answer[~valid_start] = 0
    # Bounds crossing is independent of the half-cell nearest-node convention.
    exit_t = torch.where(delta > 0, (shape - 1 - start) / delta,
                         torch.where(delta < 0, -start / delta, torch.full_like(delta, float("inf"))))
    outside_end = ((b < low) | (b > high)).any(1)
    answer = torch.where(outside_end & valid_start, exit_t.min(1).values.clamp(0, 1), answer)

    def check_points(t, active):
        position = start + delta * t[:, None]
        base = torch.floor(position + 0.5).long()
        near_face = torch.abs(position + 0.5 - torch.round(position + 0.5)) <= tol
        base = torch.where(near_face, torch.round(position + 0.5).long(), base)
        # All eight combinations matter at a corner, even if the path merely
        # touches a cell and does not spend positive time in its interior.
        bad = torch.zeros(len(a), dtype=torch.bool, device=device)
        for bits in product((0, 1), repeat=3):
            offset = torch.tensor(bits, device=device)
            enabled = ((offset == 0) | near_face).all(1)
            cell = base - offset
            inside = ((cell >= 0) & (cell < shape)).all(1)
            safe = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), shape - 1)
            sampled = labels[safe[:, 0], safe[:, 1], safe[:, 2]]
            bad |= active & enabled & (~inside | (sampled != roots))
        return bad

    current_t = torch.zeros(len(a), device=device, dtype=torch.float64)
    answer[check_points(current_t, valid_start)] = 0
    # Next face strictly beyond the origin. Starting on a face has already
    # checked both cells, so advancing must not repeatedly visit that face.
    cell = torch.floor(start + 0.5).long()
    face = cell + torch.where(step > 0, 0.5, -0.5)
    next_t = torch.where(step != 0, (face - start) / delta, torch.full_like(delta, float("inf")))
    increment = torch.where(step != 0, 1 / torch.abs(delta), torch.full_like(delta, float("inf")))
    next_t = torch.where(next_t <= tol, next_t + increment, next_t)
    # At most sum(shape) grid faces can be crossed before leaving the box.
    for _ in range(int(shape.sum().item()) + 3):
        event_t = next_t.min(1).values
        active = valid_start & (event_t <= 1) & (event_t < answer)
        if not active.any():
            break
        safe_t = torch.where(active, event_t, torch.zeros_like(event_t))
        bad = check_points(safe_t, active)
        answer = torch.where(bad, event_t, answer)
        tied = torch.abs(next_t - event_t[:, None]) <= tol
        next_t = torch.where(tied, next_t + increment, next_t)
    end_t = torch.ones(len(a), device=device, dtype=torch.float64)
    bad_end = check_points(end_t, valid_start & torch.isinf(answer))
    answer[bad_end] = 1
    return answer


def constrain_volume_segments(origins, ends, root_labels, labels, low, high):
    """Retreat a metric epsilon before first contact, without wall sliding."""
    hit = first_invalid_segment_fraction(origins, ends, root_labels, labels, low, high)
    invalid = torch.isfinite(hit)
    delta = ends - origins
    shape = torch.tensor(labels.shape, device=origins.device)
    spacing = (torch.as_tensor(high, device=origins.device).reshape(3)
               - torch.as_tensor(low, device=origins.device).reshape(3)) / (shape - 1)
    retreat = 1e-4 * spacing.min() / torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
    t = torch.where(invalid, (hit - retreat).clamp(0, 1), torch.ones_like(hit))
    corrected = origins + t.to(origins.dtype)[:, None] * delta
    return corrected, invalid


def audit_volume_strands(strands, root_labels, labels, low, high, chunk_size=32768):
    """Audit final polyline segments, reporting identities without dropping any."""
    strands = np.asarray(strands)
    if strands.ndim != 3 or strands.shape[2] != 3 or strands.shape[1] < 2:
        raise ValueError("Expected N-by-S-by-3 strands")
    roots = np.asarray(root_labels)
    if roots.shape != (len(strands),):
        raise ValueError("Root identities must survive all postprocessing")
    starts, ends = strands[:, :-1].reshape(-1, 3), strands[:, 1:].reshape(-1, 3)
    expanded = np.repeat(roots, strands.shape[1] - 1)
    illegal = np.zeros(len(starts), bool)
    label_t = torch.as_tensor(labels.astype(np.int64))
    for begin in range(0, len(starts), chunk_size):
        stop = begin + chunk_size
        hits = first_invalid_segment_fraction(torch.as_tensor(starts[begin:stop]),
                    torch.as_tensor(ends[begin:stop]), torch.as_tensor(expanded[begin:stop]),
                    label_t, low, high)
        illegal[begin:stop] = torch.isfinite(hits).numpy()
    illegal = illegal.reshape(len(strands), -1)
    locations = np.argwhere(illegal)
    return {"passed": not bool(illegal.any()), "strands": len(strands),
            "segments": int(illegal.size), "invalid_segments": int(illegal.sum()),
            "invalid_strands": int(illegal.any(1).sum()),
            "invalid_locations": locations.tolist()}
