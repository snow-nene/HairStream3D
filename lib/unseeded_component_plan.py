"""Evidence acquisition plan for unseeded volume components."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def plan_unseeded_component_evidence(domain, partition_labels, seed_labels, *, max_views=4):
    """Describe evidence still required; never assign synthetic partition labels."""
    domain = np.asarray(domain, bool)
    labels = np.asarray(partition_labels, int)
    seeds = np.asarray(seed_labels, int)
    if domain.ndim != 3 or labels.shape != domain.shape or seeds.shape != domain.shape:
        raise ValueError("domain, partitions and seeds must share XYZ shape")
    # A propagated seed volume may contain zeros at surface-unresolved cells,
    # while partition_labels is the authoritative post-propagation coverage.
    missing = domain & (labels <= 0)
    components, count = ndimage.label(missing, structure=ndimage.generate_binary_structure(3, 1))
    records = []
    for component_id in range(1, count + 1):
        mask = components == component_id
        if not mask.any():
            continue
        coordinates = np.argwhere(mask)
        records.append({"component_id": int(component_id), "voxels": int(mask.sum()),
                        "center_index": coordinates.mean(0).tolist(),
                        "required_evidence": ["mesh_visible_intersection", "strand_depth_observation"],
                        "fallback_evidence": ["additional_registered_view", "manual_curve_constraint"],
                        "max_views": int(max_views), "status": "awaiting_evidence"})
    return {"unseeded_voxels": int(missing.sum()), "component_count": len(records),
            "components": records, "formal_bundle_allowed": len(records) == 0,
            "policy": "do_not_assign_labels_without_registered evidence"}
