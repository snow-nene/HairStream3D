#!/usr/bin/env python3
"""为体积 bundle 生成物理尺度稳定的有效块与缓存摘要。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from lib.recon_strategy.volume_partition import load_bundle
from lib.spatial_budget import active_blocks, cache_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    arrays, metadata = load_bundle(args.bundle)
    domain = np.asarray(arrays["domain_mask"], bool)
    spacing = np.asarray(metadata["spacing"], float)
    config = {"block_size": args.block_size, "axis_order": metadata.get("axis_order"),
              "units": metadata.get("units"), "bundle_version": metadata.get("version")}
    blocks = active_blocks(domain, spacing, args.block_size)
    report = {"version": 1, "bundle": str(args.bundle.resolve()), "shape": list(domain.shape),
              "spacing": spacing.tolist(), "block_size": args.block_size,
              "active_blocks": blocks, "active_voxels": int(domain.sum()),
              "cache_fingerprint": cache_fingerprint(domain, spacing, config), "config": config}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("shape", "active_voxels", "cache_fingerprint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
