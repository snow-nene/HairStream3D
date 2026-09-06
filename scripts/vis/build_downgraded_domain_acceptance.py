#!/usr/bin/env python3
"""对无直接证据的体素组件执行显式 unknown/excluded 降级验收。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.coverage) as d:
        labels = np.asarray(d["partition_labels"], np.int32)
        domain = np.asarray(d["domain_mask"], bool)
    missing = domain & (labels <= 0)
    st = np.zeros((3, 3, 3), np.int8)
    st[1, 1, 1] = 1
    st[0, 1, 1] = st[2, 1, 1] = 1
    st[1, 0, 1] = st[1, 2, 1] = 1
    st[1, 1, 0] = st[1, 1, 2] = 1
    components, count = label(missing, structure=st)
    trusted_domain = domain & ~missing
    downgraded = np.zeros_like(labels)
    downgraded[trusted_domain] = labels[trusted_domain]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "downgraded_partition_domain.npz",
                        partition_labels=downgraded,
                        trusted_domain=trusted_domain,
                        excluded_unknown=missing,
                        component_ids=components)
    report = {
        "version": 1,
        "policy": "exclude_unresolved_unknown_components",
        "source_coverage": str(args.coverage.resolve()),
        "original_domain_voxels": int(domain.sum()),
        "trusted_domain_voxels": int(trusted_domain.sum()),
        "excluded_unknown_voxels": int(missing.sum()),
        "excluded_unknown_components": int(count),
        "trusted_covered_voxels": int(np.count_nonzero(trusted_domain & (downgraded > 0))),
        "trusted_coverage_fraction": 1.0 if trusted_domain.sum() else 0.0,
        "formal_bundle_allowed": bool(np.all(downgraded[trusted_domain] > 0)),
        "full_domain_acceptance": False,
        "limitations": [
            "excluded components remain unknown and cannot be used for PDE propagation",
            "this report is not equivalent to full original-domain coverage",
            "128^3/10k remains gated on this policy and must report excluded unknown volume",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
