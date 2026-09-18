#!/usr/bin/env python3
"""合并共享同一基线的补生长分支，重新审计全部线段后导出。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.template_identity import load_template_identity, require_bound_template
from scripts.recon_3d.grow_visible_surface_gaps import audit_strands, pack_supplemental_strands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    identity=load_template_identity(a.head)
    with np.load(a.base) as data:
        require_bound_template(data,Path(identity['template_path']))
        base=data['strands']
        flags=data.get('is_supplemental',np.zeros(len(base),bool))
        counts=data.get('valid_point_counts',np.full(len(base),base.shape[1]))
    extra=[]
    hashes=set()
    for run in a.runs:
        with np.load(run) as data:
            require_bound_template(data,Path(identity['template_path']))
            strands=data['strands']
            if len(strands)<len(base) or not np.array_equal(strands[:len(base),:base.shape[1]],base):
                raise ValueError('growth run does not preserve the shared base')
            for strand in strands[len(base):]:
                strand=strand[np.r_[True,np.linalg.norm(np.diff(strand,axis=0),axis=1)>0]]
                key=hashlib.sha256(strand.tobytes()).digest()
                if key not in hashes:
                    extra.append(strand)
                    hashes.add(key)
    packed=pack_supplemental_strands(base,extra)
    with np.load(a.head) as data:
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),o3d.utility.Vector3iVector(data['faces']))
    audit=audit_strands(packed,mesh)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    report={'base':str(a.base),'runs':[str(r) for r in a.runs],
            'added_count':len(extra),'collision_audit':audit,'status':'requires_fixed_multiview_audit'}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    if not audit['passed']:
        raise ValueError('combined collision audit blocked export')
    np.savez_compressed(a.output_dir/'all_root_prefixes.npz',strands=packed,
        valid_point_counts=np.r_[counts,[len(s) for s in extra]],
        is_supplemental=np.r_[flags,np.ones(len(extra),bool)],
        template_blend_sha256=np.array(identity['template_sha256']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
