"""Fixed-input scenario matrix for reproducible PDE/growth ablations."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    budget: float
    root_fingerprint: str
    configuration: dict


def scenario_matrix(root_fingerprint, *, seed, budget, configurations):
    if not root_fingerprint or seed < 0 or budget <= 0 or not configurations:
        raise ValueError("invalid fixed-input scenario matrix")
    return [Scenario(str(name), int(seed), float(budget), str(root_fingerprint), dict(config))
            for name, config in configurations.items()]


def scenario_record(scenario, metrics, *, status="complete", failure_reason=None):
    payload = {"scenario": asdict(scenario), "metrics": metrics, "status": status,
               "failure_reason": failure_reason}
    payload["record_fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def validate_matrix(records):
    if not records:
        raise ValueError("empty scenario records")
    scenarios = [record["scenario"] for record in records]
    roots = {item["root_fingerprint"] for item in scenarios}
    seeds = {item["seed"] for item in scenarios}
    budgets = {item["budget"] for item in scenarios}
    return {"fixed_roots": len(roots) == 1, "fixed_seed": len(seeds) == 1,
            "fixed_budget": len(budgets) == 1, "all_recorded": all("record_fingerprint" in r for r in records)}
