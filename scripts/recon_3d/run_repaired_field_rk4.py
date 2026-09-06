#!/usr/bin/env python3
"""用修复后的分区方向场生成完整 RK4 中间段并运行正式 guard。"""
import argparse,json
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_segment_guard import constrain_volume_segments

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--field',type=Path,required=True);ap.add_argument('--roots',type=Path,required=True);ap.add_argument('--bounds',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--steps',type=int,default=64);ap.add_argument('--step-m',type=float,default=.003);a=ap.parse_args()
 f=np.load(a.field); field=f['field']; labels=f['partition_labels']; b=np.load(a.bounds); lo,hi=b['b_min'],b['b_max']; roots=np.load(a.roots)['roots_world']; shape=np.array(labels.shape)
 def query(p):
  ix=np.rint((p-lo)/(hi-lo)*(shape-1)).astype(int); ok=((ix>=0)&(ix<shape)).all(1); ix=np.clip(ix,0,shape-1); v=field[:,ix[:,0],ix[:,1],ix[:,2]].T; lab=labels[tuple(ix.T)]; v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8); return v,lab,ok
 _,part,ok=query(roots); keep=ok&(part>0); roots=roots[keep]; part=part[keep]; strands=np.zeros((len(roots),a.steps+1,3),np.float32); strands[:,0]=roots
 for s in range(a.steps):
  p=strands[:,s]; k1,l1,o1=query(p); k2,l2,o2=query(p+.5*a.step_m*k1); k3,l3,o3=query(p+.5*a.step_m*k2); k4,l4,o4=query(p+a.step_m*k3); nxt=p+a.step_m*(k1+2*k2+2*k3+k4)/6; strands[:,s+1]=nxt
 origins=strands[:,:-1].reshape(-1,3); ends=strands[:,1:].reshape(-1,3); expanded=np.repeat(part,a.steps); corrected,invalid=constrain_volume_segments(torch.tensor(origins),torch.tensor(ends),torch.tensor(expanded),torch.tensor(labels),lo,hi); guarded=corrected.numpy().reshape(strands.shape[0],a.steps,3); strands[:,1:]=guarded
 a.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output_dir/'repaired_rk4_intermediate.npz',strands=strands,root_labels=part,invalid_segments=invalid.numpy().reshape(len(roots),a.steps)); report={'roots_input':int(len(roots)),'steps':a.steps,'step_m':a.step_m,'segments':int(len(origins)),'guarded_segments':int(invalid.sum()),'exportable_roots':int((~invalid.numpy().reshape(len(roots),a.steps).any(1)).sum()),'policy':'production_repaired_field_rk4_with_volume_segment_guard'}; (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
