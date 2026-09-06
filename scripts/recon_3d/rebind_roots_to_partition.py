#!/usr/bin/env python3
"""按当前 volume bundle 重新绑定根点 partition 标签。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--roots',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 r=np.load(a.roots); roots=np.asarray(r['roots_world'],float); b=np.load(a.bundle); dom=b['domain_mask']; labels=b['partition_labels']; q=np.load('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/mesh_front_64/candidate_domain_seeds.npz'); lo,hi=q['b_min'],q['b_max']; ix=np.rint((roots-lo)/(hi-lo)*(np.array(dom.shape)-1)).astype(int); inside=((ix>=0)&(ix<np.array(dom.shape))).all(1); ix=np.clip(ix,0,np.array(dom.shape)-1); lab=labels[tuple(ix.T)]; valid=inside&dom[tuple(ix.T)]&(lab>0); reason=np.where(~inside,'outside_grid',np.where(~dom[tuple(ix.T)],'outside_trusted_domain',np.where(lab<=0,'missing_partition','accepted'))); a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,roots_world=roots[valid],partition_labels=lab[valid],source_indices=np.flatnonzero(valid)); rep={'total_roots':len(roots),'accepted_roots':int(valid.sum()),'rejected_roots':int((~valid).sum()),'rejection_reasons':{str(x):int((reason==x).sum()) for x in np.unique(reason) if x!='accepted'}}; a.output.with_suffix('.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)); print(json.dumps(rep,ensure_ascii=False))
if __name__=='__main__': main()
