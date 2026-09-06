"""Deterministic correspondence of local 2-D partition labels across views."""
from __future__ import annotations

import numpy as np


def correlate_partition_labels(view_labels, projected_pixels, *, min_overlap=1):
    """Map local labels to global ids using weighted pixel overlap.

    Ties and contradictory dominant labels are retained in ``conflicts``;
    they are never resolved by label-number order.
    """
    if len(view_labels) != len(projected_pixels):
        raise ValueError("view labels and projections must have equal length")
    scores = {}
    for view, (labels, pixels) in enumerate(zip(view_labels, projected_pixels)):
        labels = np.asarray(labels, int).reshape(-1)
        pixels = np.asarray(pixels, int).reshape(-1, 2)
        if len(labels) != len(pixels):
            raise ValueError("labels and pixels must match")
        for label, pixel in zip(labels, pixels):
            if label <= 0:
                continue
            key = (int(pixel[0]), int(pixel[1]))
            scores.setdefault(key, []).append((view, int(label)))
    mapping = {}
    conflicts = []
    next_id = 1
    for pixel in sorted(scores):
        observations = scores[pixel]
        counts = {}
        for view, label in observations:
            counts.setdefault(label, set()).add(view)
        ranked = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))
        if not ranked or len(ranked[0][1]) < min_overlap:
            continue
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            conflicts.append({"pixel": pixel, "labels": [item[0] for item in ranked]})
            continue
        local = ranked[0][0]
        mapping[(pixel, local)] = next_id
        next_id += 1
    return {"mapping": mapping, "conflicts": conflicts, "observations": scores}
