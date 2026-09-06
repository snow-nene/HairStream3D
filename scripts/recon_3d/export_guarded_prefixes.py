#!/usr/bin/env python3
"""保留 RK4 发丝的合法前缀并显式记录终止原因。"""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--intermediate',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args(); d=np.load(a.intermediate); strands=np.asarray(d['strands']); invalid=np.asarray(d['invalid_segments'],bool); roots=np.asarray(d['root_labels']); n,s,_=strands.shape; truncated=invalid.any(1); term=np.where(truncated,invalid.argmax(1)+1,s-1); prefixes=np.zeros_like(strands); prefixes[:,0]=strands[:,0]
 for i in range(n):
  if truncated[i]:
   stop=max(int(term[i])-1,0); prefixes[i,:stop+1]=strands[i,:stop+1]; prefixes[i,stop+1:]=strands[i,stop][None,:]
  else: prefixes[i]=strands[i]
 reason=np.where(truncated,'partition_interface','completed'); a.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output_dir/'guarded_prefixes.npz',strands=prefixes,root_labels=roots,termination_step=term,termination_reason=reason,complete=~truncated); report={'roots':int(n),'complete_roots':int((~truncated).sum()),'truncated_roots':int(truncated.sum()),'termination_reasons':{'partition_interface':int(truncated.sum()),'completed':int((~truncated).sum())},'policy':'preserve_legal_prefix_terminate_first_guard','silent_deletion':False}; (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
