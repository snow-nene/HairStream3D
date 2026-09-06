#!/usr/bin/env python3
"""对单一 volume partition 做独立 screened-Poisson 求解。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--fusion-debug", type=Path, required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    b = np.load(args.bundle); f = np.load(args.fusion_debug)
    domain = np.asarray(b["domain_mask"], bool) & (np.asarray(b["partition_labels"]) == args.partition)
    fused = np.asarray(f["fused_orien"], np.float32)
    if fused.shape[1:] != domain.shape:
        raise ValueError("fusion orientation and bundle grids differ")
    boundary = domain & ~ndimage.binary_erosion(domain, structure=ndimage.generate_binary_structure(3, 1))
    valid = boundary & (np.linalg.norm(fused, axis=0) > 1e-6)
    values = np.zeros_like(fused)
    values[:, valid] = fused[:, valid]
    values[:, valid] /= np.maximum(np.linalg.norm(values[:, valid], axis=0, keepdims=True), 1e-8)
    field, metrics = solve_weighted_screened_poisson(
        domain, values, valid, spacing=np.ones(3, np.float32),
        tolerance=1e-4, max_iterations=5000, partition_labels=np.where(domain, 1, 0),
        require_component_convergence=True, validate_components=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "partition_field.npz", field=field, domain=domain, boundary=valid)
    report = metrics.to_dict()
    report.update({"partition": args.partition, "domain_voxels": int(domain.sum()), "boundary_voxels": int(valid.sum())})
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
