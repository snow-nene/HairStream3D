#!/usr/bin/env python3
"""运行固定输入的合成消融矩阵并保存可比较 manifest。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from lib.ablation_manifest import build_ablation_manifest
from lib.direction_orientation import classify_multiflow_ambiguity
from lib.scenario_matrix import scenario_matrix, scenario_record, validate_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--sensitivity-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--sensitivity-budgets", type=float, nargs="+", default=None)
    args = parser.parse_args()
    roots = np.arange(32, dtype=np.float32).reshape(-1, 1)
    root_fingerprint = build_ablation_manifest("synthetic-roots", seed=args.seed, budget=args.budget,
                                               configuration={"roots": len(roots)}, metrics={})["run_fingerprint"]
    configurations = {"baseline": {"mode": "straight"}, "curly": {"mode": "curly"},
                      "front_back": {"mode": "occlusion"}, "cross_flow": {"mode": "multiflow"},
                      "feedback_failure": {"mode": "rollback"}}
    scenarios = scenario_matrix(root_fingerprint, seed=args.seed, budget=args.budget, configurations=configurations)
    records = []
    for scenario in scenarios:
        vectors = np.tile([1., 0., 0.], (4, 1))
        if scenario.configuration["mode"] == "multiflow":
            vectors[1] = [0., 1., 0.]
        ambiguity = classify_multiflow_ambiguity(vectors, [1, 1, 1, 1])
        metrics = {"ambiguous_samples": int(ambiguity["ambiguous"].sum()),
                   "root_count": len(roots), "budget": args.budget}
        records.append(scenario_record(scenario, metrics, status="complete"))
    seeds = args.sensitivity_seeds or [args.seed]
    budgets = args.sensitivity_budgets or [args.budget]
    sensitivity = [{"seed": int(seed), "budget": float(budget),
                    "input_fingerprint": root_fingerprint}
                   for seed in seeds for budget in budgets]
    report = {"version": 1, "records": records, "validation": validate_matrix(records),
              "parameter_sensitivity": {"seed": args.seed, "budget": args.budget,
                                         "grid": sensitivity}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
