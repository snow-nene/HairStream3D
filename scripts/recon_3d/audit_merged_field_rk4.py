#!/usr/bin/env python3
"""对合并分区方向场执行轻量整线段 RK4 审计。"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--field',type=Path,required=True);ap.add_argument('--roots',type=Path,required=True);ap.add_argument('--bounds',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=64);ap.add_argument('--step-m',type=float,default=.003);a=ap.parse_args()
 f=np.load(a.field); field=f['field']; dom=f['domain']; labels=f['partition_labels']; b=np.load(a.bounds); lo,hi=b['b_min'],b['b_max']; r=np.load(a.roots)['roots_world']; shape=np.array(dom.shape)
 def q(p):
  ix=np.rint((p-lo)/(hi-lo)*(shape-1)).astype(int); ok=((ix>=0)&(ix<shape)).all(1); ix=np.clip(ix,0,shape-1); v=field[:,ix[:,0],ix[:,1],ix[:,2]].T; return v,labels[tuple(ix.T)],ok
 pos=r.copy(); initial,part,ok=q(pos); active=ok&(part>0)&(np.linalg.norm(initial,axis=1)>1e-6); invalid=np.zeros(len(r),bool); crossings=np.zeros(len(r),bool)
 for _ in range(a.steps):
  k1,l1,o1=q(pos); k1/=np.maximum(np.linalg.norm(k1,axis=1,keepdims=True),1e-8); k2,l2,o2=q(pos+.5*a.step_m*k1); k2/=np.maximum(np.linalg.norm(k2,axis=1,keepdims=True),1e-8); k3,l3,o3=q(pos+.5*a.step_m*k2); k3/=np.maximum(np.linalg.norm(k3,axis=1,keepdims=True),1e-8); k4,l4,o4=q(pos+a.step_m*k3); k4/=np.maximum(np.linalg.norm(k4,axis=1,keepdims=True),1e-8); nxt=pos+a.step_m*(k1+2*k2+2*k3+k4)/6; vn,ln,on=q(nxt); bad=active&(~o1|~o2|~o3|~o4|~on|(ln!=part)|(np.linalg.norm(vn,axis=1)<1e-6)); invalid|=bad; crossings|=active&(ln!=part); active&=~bad; pos[numpy_active] if False else pos
  pos[active]=nxt[active]
 report={'roots':len(r),'active_final':int(active.sum()),'invalid_roots':int(invalid.sum()),'partition_crossing_roots':int(crossings.sum()),'steps':a.steps,'step_m':a.step_m,'rk4_ready':bool(not invalid.any() and not crossings.any())}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
