"""High-confidence guide filtering and root/partition compatibility checks."""
from __future__ import annotations

import numpy as np


def filter_guides(guides, root_labels, sample_labels, confidence, *, min_confidence=.5):
    roots = np.asarray(root_labels, int).reshape(-1)
    confidence = np.asarray(confidence, float).reshape(-1)
    if len(guides) != len(roots) or len(guides) != len(confidence):
        raise ValueError("guide arrays must match")
    accepted, rejected = [], []
    for index, (guide, root, score) in enumerate(zip(guides, roots, confidence)):
        points = np.asarray(guide, float)
        labels = np.asarray(sample_labels[index], int).reshape(-1)
        reason = None
        if root <= 0:
            reason = "unrooted"
        elif points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            reason = "incomplete_trajectory"
        elif not np.isfinite(points).all() or not np.isfinite(score):
            reason = "nonfinite"
        elif score < min_confidence:
            reason = "low_confidence"
        elif len(labels) != len(points) or np.any(labels != root):
            reason = "cross_partition"
        record = {"guide_id": int(index), "root_partition": int(root), "confidence": float(score), "reason": reason}
        (rejected if reason else accepted).append(record)
    return {"accepted": accepted, "rejected": rejected, "accepted_ids": [x["guide_id"] for x in accepted]}
