"""Physical-scale-stable sparse block budgeting and cache fingerprints."""
from __future__ import annotations

import hashlib
import json
import numpy as np


def active_blocks(domain_mask, spacing, block_size=16):
    domain = np.asarray(domain_mask, bool)
    spacing = np.asarray(spacing, float)
    if domain.ndim != 3 or spacing.shape != (3,) or np.any(spacing <= 0) or block_size < 1:
        raise ValueError("invalid domain, spacing or block size")
    shape = np.ceil(np.asarray(domain.shape) / block_size).astype(int)
    blocks = []
    for index in np.ndindex(*shape):
        slices = tuple(slice(i * block_size, min((i + 1) * block_size, n))
                       for i, n in zip(index, domain.shape))
        count = int(domain[slices].sum())
        if count:
            blocks.append({"index": tuple(int(i) for i in index), "voxels": count,
                           "physical_extent_m": (np.asarray([s.stop - s.start for s in slices]) * spacing).tolist()})
    return blocks


def cache_fingerprint(domain_mask, spacing, config):
    domain = np.asarray(domain_mask, bool)
    payload = {"shape": list(domain.shape), "spacing": np.asarray(spacing, float).tolist(), "config": config,
               "domain_sha256": hashlib.sha256(domain.tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
