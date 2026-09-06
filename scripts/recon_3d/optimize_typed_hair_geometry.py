#!/usr/bin/env python3
"""HairLRM 思路的非训练实验：目标网格几何、类型化表面和 front 反馈。"""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import torch
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.compare_head_guard_integration import HeadSurface,measure,normalize
from scripts.recon_3d.recover_boundary_strands import grow,export_all
from scripts.recon_3d.recon3D import load_calib
from scripts.vis.audit_volume_head_visibility import visible_raster_metrics
from lib.recon_strategy.volume_segment_guard import audit_volume_strands
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson
from lib.mesh_observation_geometry import audit_orthographic_camera


def geometry_at_nodes(low,high,shape,head,hair,wrap):
    """真正查询目标分辨率各节点，绝不放大旧 SDF。"""
    axes=[np.linspace(low[i],high[i],shape[i]) for i in range(3)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3).astype(np.float32)
    head_scene=HeadSurface(head)
    hs=o3d.t.geometry.RaycastingScene();hs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hair))
    ws=o3d.t.geometry.RaycastingScene();ws.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wrap))
    hdist=[];hnorm=[];odist=[];onorm=[];wsdf=[]
    for start in range(0,len(points),131072):
        p=points[start:start+131072];d,n=head_scene.query(p);hdist.append(d);hnorm.append(n)
        q=hs.compute_closest_points(o3d.core.Tensor(p));odist.append(np.linalg.norm(p-q['points'].numpy(),axis=1));onorm.append(q['primitive_normals'].numpy())
        wsdf.append(ws.compute_signed_distance(o3d.core.Tensor(p),nsamples=3).numpy())
    return {'head_normal_distance':np.concatenate(hdist).reshape(shape),'head_normals':np.concatenate(hnorm).reshape(*shape,3).transpose(3,0,1,2),
            'hair_surface_distance':np.concatenate(odist).reshape(shape),'hair_surface_normals':np.concatenate(onorm).reshape(*shape,3).transpose(3,0,1,2),
            'outer_wrap_sdf':np.concatenate(wsdf).reshape(shape),'b_min':low,'b_max':high},points


def project_front_constraints(points,camera,seg,strand,head_mesh,reference):
    """只在头部前方使用 front 约束；遮挡区不当作背景。"""
    h,w=seg.shape;clip=np.c_[points,np.ones(len(points))]@camera.T;uv=(clip[:,:2]+1)*[.5*(w-1),.5*(h-1)]
    ix=np.rint(uv).astype(int);inside=((ix>=[0,0])&(ix<[w,h])).all(1);safe=np.clip(ix,[0,0],[w-1,h-1]);pix=safe[:,1]*w+safe[:,0]
    inverse=np.linalg.inv(camera);yy,xx=np.mgrid[:h,:w];grid=np.c_[2*xx.ravel()/(w-1)-1,2*yy.ravel()/(h-1)-1]
    near_z=float((np.c_[np.asarray(head_mesh.vertices),np.ones(len(head_mesh.vertices))]@camera.T)[:,2].max()+1)
    near=np.c_[grid,np.full(h*w,near_z),np.ones(h*w)]@inverse.T;ray=-inverse[:3,2];ray/=np.linalg.norm(ray)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head_mesh));hits=scene.cast_rays(o3d.core.Tensor(np.c_[near[:,:3],np.tile(ray,(h*w,1))].astype(np.float32)))['t_hit'].numpy()
    zhead=np.full(h*w,-np.inf);valid=np.isfinite(hits);world=near[valid,:3]+hits[valid,None]*ray;zhead[valid]=(np.c_[world,np.ones(len(world))]@camera.T)[:,2]
    visible=inside&(clip[:,2]>=zhead[pix]+.002*np.linalg.norm(camera[2,:3]))
    signed=ndimage.distance_transform_edt(seg)-ndimage.distance_transform_edt(~seg);dy,dx=np.gradient(ndimage.gaussian_filter(signed,1.))
    image_distance=signed[safe[:,1],safe[:,0]];image_hair=seg[safe[:,1],safe[:,0]]&inside
    rgb=strand[safe[:,1],safe[:,0]].astype(float)/255
    obs2=np.c_[1-2*rgb[:,2],2*rgb[:,1]-1];obs2=normalize(obs2)
    projected=reference@camera[:3,:3].T;sign=np.where(np.sum(obs2*projected[:,:2],axis=1)<0,-1,1)
    obs2*=sign[:,None]
    # 沿图像法线向内部的软修正，保留相机深度方向分量。
    inward=normalize(np.c_[dx[safe[:,1],safe[:,0]],dy[safe[:,1],safe[:,0]]])
    chosen=obs2.copy();edge=visible&(image_distance<6)&(image_distance>-12)
    tangent=chosen-np.sum(chosen*inward,axis=1)[:,None]*inward
    chosen[edge]=normalize(tangent[edge]+.3*inward[edge])
    obs3=np.c_[chosen*np.linalg.norm(projected[:,:2],axis=1)[:,None],projected[:,2]]@inverse[:3,:3].T
    return normalize(obs3),visible,image_hair,image_distance,edge


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--geometry-domain',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
    b=args.data_dir;base=b/'pde_governance/volume_partition_integration';out=args.output_dir
    if not out.resolve().is_relative_to(base.resolve()):raise ValueError('输出必须位于该 image 的任务目录')
    out.mkdir(parents=True,exist_ok=False)
    paths={'field':base/'boundary_tangent_pde_20260906/field.npz','roots':base/'mesh_front_128_partition_interface_repaired_iter4/rebound_roots.npz','bundle':base/'mesh_front_128_downgraded/trusted_partition_bundle_128.npz','bounds':base/'mesh_front_64/candidate_domain_seeds.npz','head':Path('data/head_model.obj'),'hair':b/'pixal3d/hair_mesh_aligned_best.obj','wrap':b/'pde_governance/volume_domain_audit/step_04_root_ray_consistency/full_model_outer_wrap_large.off','camera':b/'maps/param/front_dense_silhouette.npy','seg':b/'maps/seg/front.png','strand':b/'maps/strand_map/front.png'}
    f=np.load(paths['field']);r=np.load(paths['roots']);bundle=np.load(paths['bundle']);q=np.load(paths['bounds']);labels=f['partition_labels'];reference=f['field'];domain=labels>0;lo,hi=q['b_min'],q['b_max'];shape=labels.shape;spacing=(hi-lo)/(np.array(shape)-1)
    head=o3d.io.read_triangle_mesh(str(paths['head']));hair=o3d.io.read_triangle_mesh(str(paths['hair']));wrap=o3d.io.read_triangle_mesh(str(paths['wrap']))
    if not wrap.is_watertight():raise ValueError('外包络必须水密')
    camera=load_calib(str(paths['camera'])).numpy().astype(float);camera_check=audit_orthographic_camera(camera)
    if not camera_check['passed']:raise ValueError('相机轴系检查失败')
    geo,points=geometry_at_nodes(lo,hi,shape,head,hair,wrap);head_surface=HeadSurface(head)
    root_d,_=head_surface.query(r['roots_world']);cell_radius=.5*np.linalg.norm(spacing)
    # 几何候选不是语义域。只报告重建结果，旧标签及根点保持用于固定对照。
    geometric_cells=(geo['outer_wrap_sdf']<=.003+cell_radius)&(geo['head_normal_distance']>=-cell_radius)
    geo['geometry_candidate_cells']=geometric_cells;geo['hair_semantic_support']=bundle['evidence_class']==1;geo['unknown_semantics']=bundle['evidence_class']==0
    np.savez_compressed(out/'geometry_128.npz',**geo)
    geometry_report={'resolution':list(shape),'query_mode':'fresh_mesh_query_each_target_node','physical_spacing_m':spacing.tolist(),'head_closed':head.is_watertight(),'wrap_closed':True,'root_distance_abs_max_m':float(np.abs(root_d).max()),'camera_axes':camera_check,'independent_registration':'not_available_existing_calibration_only','geometric_cells':int(geometric_cells.sum()),'trusted_cells_outside_geometry':int((domain&~geometric_cells).sum()),'geometric_cells_without_trusted_labels':int((geometric_cells&~domain).sum()),'domain_replacement':False,'geometry_is_not_hair_semantics':True}
    if args.geometry_domain:
        original_domain=domain.copy();domain=domain&geometric_cells
        labels=labels.copy();labels[~domain]=0
        reference=reference.copy();reference[:,~domain]=0
        root_ix=np.rint((r['roots_world']-lo)/spacing).astype(int)
        if not np.all(domain[tuple(root_ix.T)]):raise ValueError('新几何域拒绝原根点，停止固定根点对照')
        bundle={k:bundle[k] for k in bundle.files}
        bundle['solid_mask']=bundle['solid_mask'] | (geo['head_normal_distance'] < -cell_radius)
        bundle['outer_excluded']=geo['outer_wrap_sdf'] > .003+cell_radius
        bundle['domain_mask']=domain;bundle['partition_labels']=labels
        geometry_report['domain_replacement']=True;geometry_report['retained_domain_cells']=int(domain.sum())
        geometry_report['all_original_roots_retained']=True
        np.savez_compressed(out/'integration_domain.npz',domain_mask=domain,partition_labels=labels,solid_mask=bundle['solid_mask'],b_min=lo,b_max=hi,removed_domain=original_domain&~domain)
    (out/'geometry_report.json').write_text(json.dumps(geometry_report,indent=2));print('geometry',geometry_report,flush=True)
    seg=cv2.imread(str(paths['seg']),0)>127;strand=cv2.cvtColor(cv2.imread(str(paths['strand'])),cv2.COLOR_BGR2RGB)
    # 只对可信域点建立约束，大体积点不重复做图像采样。
    flat_domain=domain.ravel();obs,visible,hairpix,image_distance,edge=project_front_constraints(points[flat_domain],camera,seg,strand,head,reference.reshape(3,-1)[:,flat_domain].T)
    observations=np.zeros_like(reference);observations[:,domain]=obs.T
    surf=(geo['hair_surface_distance']<.008)&(geo['head_normal_distance']>.006)&domain
    weights=np.where(surf,8*np.clip(1-geo['hair_surface_distance']/.008,0,1),0).astype(np.float32)
    scores={};baseline_dir=base/'boundary_tangent_pde_20260906'
    baseline_metrics=json.loads((baseline_dir/'report.json').read_text());baseline_visibility=json.loads((baseline_dir/'visibility.json').read_text());scores['baseline']={'geometry':baseline_metrics,'visibility':baseline_visibility}
    for name,strength in [('typed_outer',0.),('front_feedback',1.)]:
        run=out/name;run.mkdir();obs_weight=np.zeros(shape,np.float32)
        use=visible&((hairpix&(image_distance>2))|edge)
        obs_weight[domain]=np.where(use,strength,0)
        soft=1+obs_weight;target=(reference+obs_weight[None]*observations)/soft[None]
        # 外表面切向仅用于远离头皮的发体边界，根部维持现有离开头皮过渡。
        affected=(weights>0)|(obs_weight>0);fixed=domain&~ndimage.binary_dilation(affected,iterations=2)
        field,metrics=solve_weighted_screened_poisson(domain,reference,fixed,soft_values=target,soft_weight=np.where(domain,soft,0),normal_penalty_weight=weights,normal_penalty_normals=geo['hair_surface_normals'],partition_labels=labels,spacing=spacing/spacing.mean(),tolerance=1e-4,max_iterations=2000,require_component_convergence=True,validate_components=True)
        np.savez_compressed(run/'field.npz',field=field,partition_labels=labels);(run/'solver.json').write_text(json.dumps(metrics.to_dict(),indent=2))
        strands,reasons,steps,recovery,angles,first_block=grow(field,labels,r['roots_world'],r['partition_labels'],lo,hi,bundle,head_surface)
        np.savez_compressed(run/'all_root_prefixes.npz',strands=strands,root_labels=r['partition_labels'],source_indices=r['source_indices'],termination_reason=reasons,termination_step=steps,recovery_steps=recovery,maximum_deviation_deg=angles,first_block_reason=first_block)
        report=measure(strands,head_surface);report['termination_reasons']=dict(zip(*[x.tolist() for x in np.unique(reasons,return_counts=True)]));report['volume_audit']=audit_volume_strands(strands,r['partition_labels'],labels,lo,hi)
        vr,im=visible_raster_metrics(strands,camera,seg,paths['head']);cv2.imwrite(str(run/'visibility.png'),im)
        (run/'report.json').write_text(json.dumps(report,indent=2));(run/'visibility.json').write_text(json.dumps(vr,indent=2))
        passed=report['volume_audit']['passed'] and report['head_inside_10um_roots']==0
        accepted=passed and vr['visible_coverage']>=baseline_visibility['visible_coverage'] and vr['visible_leakage']<=baseline_visibility['visible_leakage']
        scores[name]={'geometry':report,'visibility':vr,'geometric_audit_passed':passed,'coverage_leakage_nonregression':accepted}
        if passed:export_all(run/'all_root_prefixes.ply',strands)
        (out/'comparison.json').write_text(json.dumps(scores,indent=2));print(name,report['length_quantiles_m'],vr,'accepted',accepted,flush=True)
    manifest={'inputs':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},'root_count':len(r['roots_world']),'same_roots':True,'same_partition_ids_on_retained_domain':True,'geometry_domain_replacement':bool(args.geometry_domain),'front_is_optimization_view_not_independent_test':True,'weight_outer':8,'outer_band_m':.008,'head_separation_m':.006,'front_weight_candidates':[0,1],'acceptance':'no head/volume violation, coverage nondecreasing AND leakage nonincreasing vs baseline','not_full_hairlrm':'no learned prior; guide supplementation and validated interface reclassification not implemented'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
