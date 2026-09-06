"""Merge registered supplemental seeds without overriding trusted labels."""
from __future__ import annotations

import numpy as np


def merge_evidence_seeds(existing_labels, supplemental_labels, source_kind, source_view, *, trusted_source="mesh_visible"):
    existing = np.asarray(existing_labels, int)
    supplemental = np.asarray(supplemental_labels, int)
    if existing.shape != supplemental.shape:
        raise ValueError("seed label arrays must match")
    if not source_kind or not source_view:
        raise ValueError("supplemental seed provenance is required")
    if source_kind not in {"mesh_visible", "registered_view", "manual_curve"}:
        raise ValueError("unsupported supplemental source kind")
    result = existing.copy()
    incoming = supplemental > 0
    conflicts = incoming & (result > 0) & (result != supplemental)
    accepted = incoming & ~conflicts & (result == 0)
    result[accepted] = supplemental[accepted]
    return {"labels": result, "accepted_voxels": int(accepted.sum()),
            "conflict_voxels": int(conflicts.sum()), "source_kind": source_kind,
            "source_view": source_view, "trusted_source": trusted_source,
            "policy": "existing positive labels win; conflicts remain recorded"}
