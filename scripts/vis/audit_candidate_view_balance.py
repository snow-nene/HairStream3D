"""原图约束候选的其他视角与固定耳前像素审计。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth,load_observation_camera
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--repair-dir',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,required=True)
    a=p.parse_args()
    if not a.candidate_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出须按 Image ID 聚合')
    d=np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    head=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),o3d.utility.Vector3iVector(d['faces']))
    if not head.is_watertight():raise ValueError('头模须闭合')
    def chart(view,camera):
        seg=cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'),0)>127
        return VisibleSurfaceGrowth(camera,seg.astype(int),np.zeros((512,512,2)),head)
    front=chart('front',load_observation_camera(a.data_dir,'front'))
    guard=SubjectGuards({'front':front},side_veto=False)
    t=np.load(a.candidate_dir/'transforms.npz');old=t['saved_icp']
    report={}
    for view in ['right','back']:
        cam=load_observation_camera(a.data_dir,view)
        original=chart(view,cam)
        ids=np.flatnonzero(original.target)[::16]
        report[view]={'fixed_sampled_target_pixels':len(ids),'stride':16,'variants':{}}
        for name in ['saved_icp','fixed_icp','photo_weight_1']:
            c=chart(view,cam@np.linalg.inv(t[name]@np.linalg.inv(old)))
            hit=c.head_hit.ravel()[ids];surface=c.head_points[ids]
            toward=c.inverse[:3,2].copy();toward/=np.linalg.norm(toward)
            feasible=np.zeros(len(ids),bool)
            for offset in np.linspace(.0008,.05,100):
                points=surface+offset*toward
                codes=guard.points(points,collision=False)
                signed=front.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)),nsamples=3).numpy()
                feasible |= hit&(codes==0)&(signed>=.0004)
            report[view]['variants'][name]={'feasible':int(feasible.sum()),'infeasible_hit':int((hit&~feasible).sum()),'no_head_hit':int((~hit).sum())}
        print(view,report[view],flush=True)
    e=np.load(a.candidate_dir/'feasibility/evidence.npz')
    local=np.load(a.repair_dir/'local_registration_audit_20260911/paired_points.npz')
    report['local_fixed_pixels']={}
    for region in ['额角','太阳穴','耳前']:
        ids=local[region+'_pixel_ids'];keep=np.isin(e['pixel_indices'],ids)
        report['local_fixed_pixels'][region]={'pixels':len(ids)}
        for name in ['saved','fixed_indices','photo_constrained']:
            hit=e[name+'_hit'][keep];ok=e[name+'_feasible'][keep].any(1)
            report['local_fixed_pixels'][region][name]={'feasible':int(ok.sum()),'infeasible_hit':int((hit&~ok).sum()),'no_head_hit':int((~hit).sum())}
    report['limits']='右/后为原目标区域每16个像素抽样，不是完整渲染缺口；相机随GLB变换同步更新，front固定。可行性按0.8–50mm的100个偏移采样。'
    (a.candidate_dir/'view_balance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
