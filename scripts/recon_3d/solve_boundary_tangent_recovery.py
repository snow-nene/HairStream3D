#!/usr/bin/env python3
"""将分区硬界面的切向条件纳入 PDE，随后按固定根点进行守卫积分。"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson
from lib.recon_strategy.volume_segment_guard import audit_volume_strands
from scripts.recon_3d.compare_head_guard_integration import HeadSurface,measure
from scripts.recon_3d.recover_boundary_strands import grow,export_all


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(4)
    if not a.output_dir.resolve().is_relative_to(a.base.resolve()):raise ValueError('输出必须属于该 Image ID 任务目录')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    fp=a.base/'mesh_front_128_partition_interface_repaired_iter4/repaired_partition_field.npz';rp=a.base/'mesh_front_128_partition_interface_repaired_iter4/rebound_roots.npz';bp=a.base/'mesh_front_128_downgraded/trusted_partition_bundle_128.npz';qp=a.base/'mesh_front_64/candidate_domain_seeds.npz'
    f=np.load(fp);r=np.load(rp);b=np.load(bp);q=np.load(qp);labels=f['partition_labels'];domain=labels>0;reference=f['field'];low,high=q['b_min'],q['b_max'];spacing=(high-low)/(np.array(labels.shape)-1)
    interface=np.zeros_like(domain)
    for axis in range(3):
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None);left=tuple(left);right=tuple(right)
        edge=domain[left]&domain[right]&(labels[left]!=labels[right]);interface[left]|=edge;interface[right]|=edge
    normal=np.zeros_like(reference);band=np.zeros_like(domain)
    for identity in np.unique(labels[domain]):
        own=labels==identity;other=domain&~own
        distance=ndimage.distance_transform_edt(~other,sampling=spacing)
        normals=np.stack(np.gradient(distance,*spacing),axis=0)
        mask=own&(distance<2.1*spacing.max());normal[:,mask]=normals[:,mask];band|=mask
    normal/=np.maximum(np.linalg.norm(normal,axis=0,keepdims=True),1e-8)
    # 固定带外方向；带内优化保真、平滑与无穿界法向分量。使用体素归一化间距并记录尺度。
    fixed=domain&~ndimage.binary_dilation(band,iterations=2)
    soft=np.where(domain,1.,0.).astype(np.float32);weight=np.where(band,30.,0.).astype(np.float32)
    field,metrics=solve_weighted_screened_poisson(domain,reference,fixed,spacing=spacing/spacing.mean(),soft_values=reference,soft_weight=soft,normal_penalty_weight=weight,normal_penalty_normals=normal,partition_labels=labels,tolerance=1e-4,max_iterations=2000,require_component_convergence=True,validate_components=True)
    np.savez_compressed(a.output_dir/'field.npz',field=field,partition_labels=labels,interface_band=band,normals=normal)
    solver=metrics.to_dict();solver['interface_band_voxels']=int(band.sum());solver['physical_spacing_m']=spacing.tolist();solver['penalty_weight']=30.;solver['same_partition_labels']=True
    (a.output_dir/'solver.json').write_text(json.dumps(solver,indent=2))
    (a.output_dir/'manifest.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fp,rp,bp,qp,Path('data/head_model.obj')]},'policy':'interface_tangent_penalty_fixed_exterior_same_roots_no_domain_expansion','budget_m':.192},indent=2))
    head=HeadSurface(o3d.io.read_triangle_mesh('data/head_model.obj'))
    strands,reasons,steps,recovery,angles,first_block=grow(field,labels,r['roots_world'],r['partition_labels'],low,high,b,head)
    np.savez_compressed(a.output_dir/'all_root_prefixes.npz',strands=strands,root_labels=r['partition_labels'],source_indices=r['source_indices'],termination_reason=reasons,termination_step=steps,recovery_steps=recovery,maximum_deviation_deg=angles,first_block_reason=first_block)
    report=measure(strands,head);report['termination_reasons']=dict(zip(*[x.tolist()for x in np.unique(reasons,return_counts=True)]));report['volume_audit']=audit_volume_strands(strands,r['partition_labels'],labels,low,high);report['render_selection']='all_nonzero_prefixes'
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['volume_audit']['passed'] or report['head_inside_10um_roots']:raise RuntimeError('审计失败，拒绝 PLY 导出')
    export_all(a.output_dir/'all_root_prefixes.ply',strands)

if __name__=='__main__':main()
