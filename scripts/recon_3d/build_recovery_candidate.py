#!/usr/bin/env python3
"""构造未知组件传播与接口重叠带的 recovery_candidate（不替换正式结果）。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--bounds', type=Path, required=True)
    ap.add_argument('--field', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--overlap-width', type=int, default=2)
    args = ap.parse_args()
    b = np.load(args.bundle); bd = np.load(args.bounds); f = np.load(args.field)
    labels = np.asarray(b['partition_labels'], np.int8)
    domain = np.asarray(b['domain_mask'], bool)
    field = np.asarray(f['field'], np.float32)
    unknown = domain & (labels == 0)
    seeded = domain & (labels > 0)
    dist, idx = ndimage.distance_transform_edt(~seeded, return_indices=True)
    propagated = labels.copy()
    propagated[unknown] = labels[tuple(idx[:, unknown])]
    propagated[~domain] = 0
    # Candidate interface: original interface plus a narrow band on both sides.
    interface = domain & (propagated == 0)
    interface |= domain & (propagated == 1) & ndimage.binary_dilation(propagated == 2, iterations=args.overlap_width)
    interface |= domain & (propagated == 2) & ndimage.binary_dilation(propagated == 1, iterations=args.overlap_width)
    out_field = field.copy()
    # Smooth only the overlap band; this is a candidate continuation, not a new authoritative solve.
    for axis in range(3):
        sm = ndimage.gaussian_filter(field[axis], sigma=1.0)
        out_field[axis, interface] = 0.5 * field[axis, interface] + 0.5 * sm[interface]
    n = np.linalg.norm(out_field, axis=0, keepdims=True)
    out_field /= np.maximum(n, 1e-8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/'recovery_candidate_field.npz', field=out_field, partition_labels=propagated, domain_mask=domain, unknown_mask=unknown, overlap_mask=interface, b_min=bd['b_min'], b_max=bd['b_max'])
    comps, count = ndimage.label(unknown)
    report = {'status':'recovery_candidate','authoritative':False,'source_bundle':str(args.bundle.resolve()),'unknown_voxels_before':int(unknown.sum()),'unknown_components_before':int(count),'propagated_unknown_voxels':int((unknown & (propagated>0)).sum()),'overlap_width_voxels':args.overlap_width,'overlap_voxels':int(interface.sum()),'policy':'nearest_authoritative_partition_propagation_plus_overlap_blend','replacement_allowed':False,'audit_required':['component_residual','view_consistency','full_10k_guard']}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
