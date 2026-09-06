#!/usr/bin/env python3
"""根据 authoritative 分区覆盖生成无种子组件补证据计划。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.unseeded_component_plan import plan_unseeded_component_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-mask", type=Path, help="传播后正分区 coverage 的 npy/npz")
    args = parser.parse_args()
    with np.load(args.bundle) as data:
        domain = np.asarray(data["domain_mask"], bool)
        partitions = np.asarray(data["partition_labels"], int)
        seeds = np.asarray(data["seeds"], int) if "seeds" in data else np.zeros_like(partitions)
    if args.coverage_mask is None and not np.any(partitions > 0):
        raise ValueError("bundle has no authoritative propagated partition coverage; pass --coverage-mask")
    if args.coverage_mask:
        loaded = np.load(args.coverage_mask)
        coverage = loaded["coverage"] if hasattr(loaded, "files") and "coverage" in loaded.files else loaded
        coverage = np.asarray(coverage, bool)
        if coverage.shape != domain.shape:
            raise ValueError("coverage mask shape mismatch")
        partitions = np.where(coverage, np.maximum(partitions, 1), 0)
    report = plan_unseeded_component_evidence(domain, partitions, seeds)
    report["bundle"] = str(args.bundle.resolve())
    report["coverage_source"] = str(args.coverage_mask.resolve()) if args.coverage_mask else "partition_labels"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"components": report["component_count"], "voxels": report["unseeded_voxels"],
                      "formal_bundle_allowed": report["formal_bundle_allowed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
