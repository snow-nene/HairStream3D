"""Reproducible ablation configuration and comparison records."""
from __future__ import annotations

import hashlib
import json


def build_ablation_manifest(input_fingerprint, *, seed, budget, configuration, metrics):
    if not input_fingerprint or int(seed) < 0 or float(budget) <= 0:
        raise ValueError("invalid ablation identity or budget")
    payload = {"input_fingerprint": str(input_fingerprint), "seed": int(seed),
               "budget": float(budget), "configuration": configuration, "metrics": metrics}
    payload["run_fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def compare_ablation_manifests(manifests):
    if not manifests:
        raise ValueError("at least one ablation manifest is required")
    inputs = {item["input_fingerprint"] for item in manifests}
    budgets = {item["budget"] for item in manifests}
    return {"same_input": len(inputs) == 1, "same_budget": len(budgets) == 1,
            "runs": [item["run_fingerprint"] for item in manifests],
            "comparable": len(inputs) == 1 and len(budgets) == 1}
