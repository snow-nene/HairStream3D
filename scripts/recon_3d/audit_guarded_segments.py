#!/usr/bin/env python3
"""用正式 volume_segment_guard 审计并截断跨分区段。"""
import argparse,json
from pathlib import Path
import sys
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_segment_guard import constrain_volume_segments

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--segments',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--bounds',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(); d=np.load(a.segments); o,e,roots=d['origins'],d['ends'],d['root_labels']; l=np.load(a.labels)['partition_labels']; b=np.load(a.bounds); c,invalid=constrain_volume_segments(torch.tensor(o),torch.tensor(e),torch.tensor(roots),torch.tensor(l),b['b_min'],b['b_max']); a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output.with_suffix('.npz'),origins=o,ends=c.numpy(),root_labels=roots,invalid=invalid.numpy()); a.output.write_text(json.dumps({'segments':len(o),'invalid_segments':int(invalid.sum()),'guard':'constrain_volume_segments','cross_partition_export_blocked':True},indent=2)); print(json.dumps({'segments':len(o),'invalid_segments':int(invalid.sum())}))
if __name__=='__main__': main()
