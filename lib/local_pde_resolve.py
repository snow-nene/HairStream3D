"""Local PDE update extraction/merge with partition and halo safeguards."""
from __future__ import annotations

import numpy as np


def extract_local_problem(domain, values, boundary, partitions, affected_mask, halo=1):
    domain = np.asarray(domain, bool)
    values = np.asarray(values)
    boundary = np.asarray(boundary, bool)
    partitions = np.asarray(partitions, int)
    affected = np.asarray(affected_mask, bool)
    if domain.ndim != 3 or values.shape != (3, *domain.shape) or any(x.shape != domain.shape for x in (boundary, partitions, affected)):
        raise ValueError("local PDE arrays must share XYZ shape")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    ids = np.argwhere(affected & domain)
    if len(ids) == 0:
        raise ValueError("affected region is empty")
    lo = np.maximum(ids.min(0) - int(halo), 0)
    hi = np.minimum(ids.max(0) + int(halo) + 1, np.asarray(domain.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return {"slices": slices, "domain": domain[slices].copy(), "values": values[(slice(None), *slices)].copy(),
            "boundary": boundary[slices].copy(), "partitions": partitions[slices].copy(),
            "affected": affected[slices].copy(), "origin": lo.astype(int)}


def merge_local_field(full_field, local_field, problem):
    full = np.asarray(full_field).copy()
    local = np.asarray(local_field)
    slices = problem["slices"]
    if full.ndim != 4 or local.shape != (3, *(problem["domain"].shape)):
        raise ValueError("local field shape mismatch")
    update = problem["affected"] & problem["domain"] & ~problem["boundary"]
    target = full[(slice(None), *slices)]
    target[:, update] = local[:, update]
    full[(slice(None), *slices)] = target
    return full
