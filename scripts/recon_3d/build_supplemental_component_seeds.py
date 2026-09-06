#!/usr/bin/env python3
"""从已有多视图表面种子生成未播种组件的可审计候选标签。

该步骤只做证据绑定和候选生成，不把最近邻推断伪装成 authoritative coverage。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=Path, required=True)
    ap.add_argument("--surface-seeds", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.coverage) as d:
        base = np.asarray(d["partition_labels"], np.int32)
        domain = np.asarray(d["domain_mask"], bool)
    if base.shape != domain.shape:
        raise ValueError("coverage labels/domain shape mismatch")

    candidates = []
    for path in args.surface_seeds:
        with np.load(path) as d:
            volume = np.asarray(d["seed_volume"], np.int32)
        if volume.shape != base.shape:
            raise ValueError(f"shape mismatch: {path}")
        candidates.append((path, volume))

    missing = domain & (base <= 0)
    structure = np.zeros((3, 3, 3), np.int8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    component_map, count = label(missing, structure=structure)
    out = np.zeros_like(base)
    provenance = []
    for cid in range(1, count + 1):
        vox = np.argwhere(component_map == cid)
        best = None
        for path, volume in candidates:
            valid = volume > 0
            if not valid.any():
                continue
            dist, inds = distance_transform_edt(~valid, return_indices=True)
            for v in vox:
                idx = tuple(v)
                src = tuple(int(x) for x in inds[(slice(None),) + idx])
                d = float(dist[idx])
                label_value = int(volume[src])
                item = (d, str(path), src, label_value)
                if best is None or item < best:
                    best = item
        if best is None:
            provenance.append({"component_id": int(cid), "voxels": int(len(vox)), "status": "no_candidate"})
            continue
        d, path, src, value = best
        out[tuple(vox.T)] = value
        provenance.append({
            "component_id": int(cid), "voxels": int(len(vox)),
            "status": "candidate", "label": int(value),
            "source_file": str(Path(path).resolve()),
            "source_voxel": list(src), "nearest_voxel_distance": d,
            "inference_mode": "nearest_surface_seed",
            "authoritative": False,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=out)
    report = {
        "version": 1,
        "coverage": str(args.coverage.resolve()),
        "surface_seed_files": [str(p.resolve()) for p, _ in candidates],
        "component_count": int(count),
        "candidate_voxels": int(np.count_nonzero(out)),
        "authoritative": False,
        "formal_bundle_allowed": False,
        "policy": "nearest surface labels are candidates only; direct visible intersection or registered curve is required for formal acceptance",
        "components": provenance,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"components": count, "candidate_voxels": int(np.count_nonzero(out)), "authoritative": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
