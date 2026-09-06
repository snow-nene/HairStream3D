#!/usr/bin/env python3
"""将网格可见 seeds 传播为 authoritative partition coverage；不填补无锚组件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_partition import propagate_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="candidate_domain_seeds.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input) as data:
        domain = np.asarray(data["domain_mask"], bool)
        seeds = np.asarray(data["seeds"], np.int32)
        confidence = np.asarray(data["evidence_confidence"], float)
        spacing = (np.asarray(data["b_max"], float) - np.asarray(data["b_min"], float)) / np.maximum(np.asarray(domain.shape) - 1, 1)
    labels = np.zeros_like(seeds, np.int32)
    partition_confidence = np.zeros(domain.shape, np.float32)
    structure = ndimage.generate_binary_structure(3, 1)
    components, count = ndimage.label(domain, structure=structure)
    for component_id in range(1, count + 1):
        local = components == component_id
        if not np.any(seeds[local] > 0):
            continue
        local_labels, local_confidence = propagate_partitions(local, np.where(local, seeds, 0), spacing, np.where(local, confidence, 0.0))
        labels[local] = local_labels[local]
        partition_confidence[local] = local_confidence[local]
    covered = labels > 0
    components = int(np.max(ndimage.label(domain & ~covered, structure=structure)[0]))
    report = {"version": 1, "source": str(args.input.resolve()), "shape": list(domain.shape),
              "domain_voxels": int(domain.sum()), "covered_voxels": int(covered.sum()),
              "unseeded_voxels": int((domain & ~covered).sum()), "unseeded_components": components,
              "authoritative": True, "formal_bundle_allowed": bool(np.all(covered[domain])),
              "policy": "preserve unanchored components as zero; require new evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, partition_labels=labels, partition_confidence=partition_confidence,
                        coverage=covered, domain_mask=domain)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
