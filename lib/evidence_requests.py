"""Evidence requests for unresolved volume components."""
from __future__ import annotations

import numpy as np


def build_component_requests(plan, b_min, b_max, shape, views=("front", "left", "right", "back")):
    low = np.asarray(b_min, float); high = np.asarray(b_max, float); shape = np.asarray(shape, int)
    if low.shape != (3,) or high.shape != (3,) or shape.shape != (3,) or np.any(high <= low):
        raise ValueError("invalid volume bounds")
    requests = []
    for item in plan.get("components", []):
        center = np.asarray(item["center_index"], float)
        world = low + center / np.maximum(shape - 1, 1) * (high - low)
        requests.append({"component_id": int(item["component_id"]), "voxels": int(item["voxels"]),
                         "center_index": center.tolist(), "center_world": world.tolist(),
                         "views": list(views), "required": ["visible_mesh_intersection", "strand_map", "depth_map"],
                         "optional": ["registered_extra_view", "manual_curve"], "status": "request_pending"})
    return {"requests": requests, "request_count": len(requests), "policy": "evidence_only_no_label_filling"}
