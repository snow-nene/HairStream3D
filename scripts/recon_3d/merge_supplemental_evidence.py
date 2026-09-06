#!/usr/bin/env python3
"""合并注册补充 seeds 并输出正式门禁前的证据报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.evidence_seed_merge import merge_evidence_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--source-kind", choices=["mesh_visible", "registered_view", "manual_curve"], required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base) as data:
        base = np.asarray(data["partition_labels"], int)
        domain = np.asarray(data["domain_mask"], bool) if "domain_mask" in data else np.ones(base.shape, bool)
    with np.load(args.supplemental) as data:
        extra = np.asarray(data["labels"] if "labels" in data else data["partition_labels"], int)
    result = merge_evidence_seeds(base, extra, args.source_kind, args.source_view)
    unresolved = int(np.count_nonzero(domain & (result["labels"] <= 0)))
    report = {"version": 1, "base": str(args.base.resolve()), "supplemental": str(args.supplemental.resolve()),
              "source_kind": args.source_kind, "source_view": args.source_view,
              "accepted_voxels": result["accepted_voxels"], "conflict_voxels": result["conflict_voxels"],
              "unresolved_voxels": unresolved, "formal_bundle_allowed": unresolved == 0,
              "policy": result["policy"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, partition_labels=result["labels"])
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
