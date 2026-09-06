"""Coverage-driven supplemental growth planning with explicit stop reasons."""
from __future__ import annotations

import numpy as np

from lib.coverage_budget import allocate_coverage_roots, coverage_gain
from lib.guide_quality import filter_guides


def plan_supplemental_growth(candidates, partitions, visible, existing_roots, target_points,
                             guides, guide_root_labels, guide_sample_labels, guide_confidence,
                             *, budget, min_confidence=.5, radius=.01, seed=42):
    guide_report = filter_guides(guides, guide_root_labels, guide_sample_labels, guide_confidence,
                                 min_confidence=min_confidence)
    roots = allocate_coverage_roots(candidates, partitions, visible, existing_roots,
                                    budget=budget, min_distance=radius, seed=seed)
    gain = coverage_gain(roots["points"], target_points, radius)
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(np.asarray(visible, bool) & (np.asarray(partitions, int) > 0))
    random_ids = rng.permutation(pool)[:len(roots["indices"])] if len(pool) else np.empty(0, int)
    random_gain = coverage_gain(np.asarray(candidates)[random_ids], target_points, radius)
    stop_reason = "budget_exhausted" if len(roots["indices"]) >= budget else "no_visible_partition_candidates"
    if not len(guide_report["accepted_ids"]):
        stop_reason = "no_usable_guides"
    return {"roots": roots, "guides": guide_report, "coverage_gain": gain,
            "random_baseline_gain": random_gain, "stop_reason": stop_reason,
            "existing_root_count": int(len(existing_roots)), "seed": int(seed)}
