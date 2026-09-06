#!/usr/bin/env python3
"""将降级域从 64³ 映射到目标分辨率（默认 128³）。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def nearest_resize(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    coords = [np.minimum((np.arange(n) * old / n).astype(int), old - 1)
              for n, old in zip(shape, mask.shape)]
    return mask[np.ix_(*coords)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted-domain", type=Path, required=True)
    ap.add_argument("--target-volume", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.trusted_domain) as src:
        trusted = np.asarray(src["trusted_domain"], bool)
        excluded = np.asarray(src["excluded_unknown"], bool)
    with np.load(args.target_volume) as tgt:
        target_domain = np.asarray(tgt["domain_mask"], bool)
        b_min = np.asarray(tgt["b_min"], float) if "b_min" in tgt else None
        b_max = np.asarray(tgt["b_max"], float) if "b_max" in tgt else None
    if trusted.shape != excluded.shape:
        raise ValueError("trusted/excluded shape mismatch")
    out_trusted = nearest_resize(trusted, target_domain.shape) & target_domain
    out_excluded = nearest_resize(excluded, target_domain.shape) & target_domain
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "trusted_domain_128.npz",
                        trusted_domain=out_trusted,
                        excluded_unknown=out_excluded,
                        target_domain=target_domain)
    report = {
        "version": 1, "source": str(args.trusted_domain.resolve()),
        "target_volume": str(args.target_volume.resolve()),
        "shape": list(target_domain.shape), "resampling": "nearest_index_physical_grid",
        "target_domain_voxels": int(target_domain.sum()),
        "trusted_domain_voxels": int(out_trusted.sum()),
        "excluded_unknown_voxels": int(out_excluded.sum()),
        "trusted_fraction": float(out_trusted.sum() / max(target_domain.sum(), 1)),
        "bounds_min": b_min.tolist() if b_min is not None else None,
        "bounds_max": b_max.tolist() if b_max is not None else None,
        "formal_bundle_allowed": bool(np.all(out_trusted[target_domain] | out_excluded[target_domain])),
        "full_domain_acceptance": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
