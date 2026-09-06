"""复跑固定场验证语义终止归因；不覆盖既有轨迹或渲染。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recover_boundary_strands import grow
from scripts.recon_3d.compare_head_guard_integration import HeadSurface


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args();run=args.run_dir;torch.set_num_threads(4)
    manifest=json.loads((run/'manifest.json').read_text())
    root_data=np.load(manifest['inputs']['roots']['path'])
    bundle_file=np.load(manifest['inputs']['bundle']['path'])
    bundle={k:bundle_file[k] for k in bundle_file.files}
    domain=np.load(run/'integration_domain.npz');geo=np.load(run/'geometry_128.npz')
    low,high=domain['b_min'],domain['b_max'];labels=domain['partition_labels']
    radius=.5*np.linalg.norm((high-low)/(np.array(labels.shape)-1))
    bundle['solid_mask']=domain['solid_mask']
    bundle['outer_excluded']=geo['outer_wrap_sdf']>.003+radius
    head=HeadSurface(o3d.io.read_triangle_mesh(manifest['inputs']['head']['path']))
    reports={}
    for name in ['typed_outer','front_feedback']:
        field=np.load(run/name/'field.npz')['field']
        s,reasons,steps,_,_,first=grow(field,labels,root_data['roots_world'],
            root_data['partition_labels'],low,high,bundle,head)
        original=np.load(run/name/'all_root_prefixes.npz')['strands']
        error=float(np.abs(s-original).max())
        if error>1e-7:raise RuntimeError('归因改动意外改变轨迹，不能沿用原渲染')
        counts=dict(zip(*[x.tolist()for x in np.unique(reasons,return_counts=True)]))
        np.savez_compressed(run/name/'typed_termination.npz',
            termination_reason=reasons,termination_step=steps,
            first_block_reason=first,source_indices=root_data['source_indices'])
        reports[name]={'termination_reasons':counts,'max_trajectory_difference_m':error,
                      'semantics_are_inherited_evidence_not_new_ground_truth':True}
    (run/'termination_attribution.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
