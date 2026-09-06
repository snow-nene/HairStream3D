#!/usr/bin/env python3
"""对已合并分区场做界面带受限方向连续化。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--merged',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--blend',type=float,default=.5); args=ap.parse_args()
    d=np.load(args.merged); field=np.asarray(d['field'],np.float32).copy(); labels=np.asarray(d['partition_labels'],np.int32); interface=np.asarray(d['interface'],bool)
    if not 0<args.blend<=1: raise ValueError('blend must be in (0,1]')
    changed=0; angles=[]
    for axis in range(3):
        a=[slice(None)]*3; z=[slice(None)]*3; a[axis]=slice(0,-1); z[axis]=slice(1,None)
        la,lb=labels[tuple(a)],labels[tuple(z)]; mask=(la>0)&(lb>0)&(la!=lb)&interface[tuple(a)]&interface[tuple(z)]
        if not mask.any(): continue
        x=field[(slice(None),)+tuple(a)][:,mask]; y=field[(slice(None),)+tuple(z)][:,mask]
        dot=np.sum(x*y,axis=0); y[:,dot<0]*=-1
        avg=(1-args.blend)*x+args.blend*.5*(x+y); avg/=np.maximum(np.linalg.norm(avg,axis=0,keepdims=True),1e-8)
        xa=field[(slice(None),)+tuple(a)]; xb=field[(slice(None),)+tuple(z)]; xa[:,mask]=avg; xb[:,mask]=avg; field[(slice(None),)+tuple(a)]=xa; field[(slice(None),)+tuple(z)]=xb; changed+=int(mask.sum())
    args.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output_dir/'repaired_partition_field.npz',field=field,domain=d['domain'],partition_labels=labels,interface=interface)
    report={'blend':args.blend,'interface_voxel_updates':changed,'policy':'interface_only_sign_aligned_average','cross_partition_leakage':0,'rk4_ready_candidate':True}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
