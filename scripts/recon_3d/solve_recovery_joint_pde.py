#!/usr/bin/env python3
"""在候选分区上执行带接口软约束的联合 screened-Poisson PDE。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--fusion',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    c=np.load(args.candidate); f=np.load(args.fusion); domain=c['domain_mask'].astype(bool); labels=c['partition_labels'].astype(np.int8); observed=f['fused_orien'].astype(np.float32)
    boundary=domain & ~ndimage.binary_erosion(domain,structure=ndimage.generate_binary_structure(3,1)); valid=boundary & (np.linalg.norm(observed,axis=0)>1e-6)
    interface=c['overlap_mask'].astype(bool) & domain; unknown=c['unknown_mask'].astype(bool) & domain
    soft_values=np.asarray(c['field'],np.float32); soft_weight=np.where(interface,2.0,0.0).astype(np.float32); soft_weight[unknown]=np.maximum(soft_weight[unknown],0.5)
    values=np.zeros_like(observed); values[:,valid]=observed[:,valid]; values[:,valid]/=np.maximum(np.linalg.norm(values[:,valid],axis=0,keepdims=True),1e-8)
    field,metrics=solve_weighted_screened_poisson(domain,values,valid,spacing=np.ones(3,np.float32),soft_values=soft_values,soft_weight=soft_weight,tolerance=1e-4,max_iterations=5000,partition_labels=labels,require_component_convergence=True,validate_components=True)
    args.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output_dir/'joint_recovery_field.npz',field=field,partition_labels=labels,domain_mask=domain,interface_mask=interface)
    report=metrics.to_dict(); report.update({'status':'recovery_candidate','solver_mode':'joint_screened_poisson_with_soft_interface_overlap','authoritative':False,'domain_voxels':int(domain.sum()),'unknown_voxels_after_propagation':0,'interface_soft_voxels':int((soft_weight>0).sum()),'unknown_soft_anchors':int(unknown.sum()),'replacement_allowed':False})
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
