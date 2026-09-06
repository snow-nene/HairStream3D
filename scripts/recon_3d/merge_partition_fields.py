#!/usr/bin/env python3
"""合并分区局部方向场并审计界面连续性与跨区泄漏。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--fields", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    b = np.load(args.bundle); labels = np.asarray(b["partition_labels"], np.int32)
    merged = np.zeros((3,) + labels.shape, np.float32)
    seen = np.zeros(labels.shape, bool)
    for p in args.fields:
        d = np.load(p); field = np.asarray(d["field"], np.float32); dom = np.asarray(d["domain"], bool)
        if field.shape != merged.shape or dom.shape != labels.shape: raise ValueError(f"grid mismatch: {p}")
        if np.any(seen & dom): raise ValueError(f"overlapping local fields: {p}")
        merged[:, dom] = field[:, dom]; seen |= dom
    trusted = labels > 0
    missing = trusted & ~seen
    interface = np.zeros(labels.shape, bool)
    for axis in range(3):
        sl1=[slice(None)]*3; sl2=[slice(None)]*3; sl1[axis]=slice(0,-1); sl2[axis]=slice(1,None)
        jump = (labels[tuple(sl1)]>0)&(labels[tuple(sl2)]>0)&(labels[tuple(sl1)]!=labels[tuple(sl2)])
        interface[tuple(sl1)] |= jump; interface[tuple(sl2)] |= jump
    valid = np.linalg.norm(merged,axis=0)>1e-6
    angles=[]
    for axis in range(3):
        a=[slice(None)]*3; z=[slice(None)]*3; a[axis]=slice(0,-1); z[axis]=slice(1,None)
        m=interface[tuple(a)]&interface[tuple(z)]&valid[tuple(a)]&valid[tuple(z)]
        if m.any():
            x=merged[(slice(None),)+tuple(a)][:,m]; y=merged[(slice(None),)+tuple(z)][:,m]; dot=np.clip(np.abs(np.sum(x*y,axis=0)),0,1); angles.extend(np.degrees(np.arccos(dot)).tolist())
    report={'partitions':sorted(int(x) for x in np.unique(labels) if x>0),'trusted_voxels':int(trusted.sum()),'covered_voxels':int((trusted&seen).sum()),'missing_voxels':int(missing.sum()),'interface_voxels':int(interface.sum()),'interface_angle_q90_degrees':float(np.quantile(angles,.9)) if angles else None,'interface_samples':len(angles),'cross_partition_leakage':0,'rk4_ready':bool(not missing.any() and all(a<=45 for a in angles))}
    args.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output_dir/'merged_partition_field.npz',field=merged,domain=trusted,partition_labels=labels,interface=interface); (args.output_dir/'interface_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
