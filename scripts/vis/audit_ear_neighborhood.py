"""耳前三维离散邻域可行性与相邻解连线审计；不修改重建输入。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth,load_observation_camera
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--repair-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出必须按Image ID聚合')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),o3d.utility.Vector3iVector(d['faces']))
    if not mesh.is_watertight():raise ValueError('须使用闭合头模')
    transforms=np.load(a.repair_dir/'photo_constrained_registration_20260911/transforms.npz')
    delta=transforms['photo_weight_1']@np.linalg.inv(transforms['saved_icp'])
    def chart(view):
        cam=load_observation_camera(a.data_dir,view)
        if view!='front':cam=cam@np.linalg.inv(delta)
        mask=cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'),0)>127
        return VisibleSurfaceGrowth(cam,mask.astype(int),np.zeros((512,512,2)),mesh)
    front,left=chart('front'),chart('left');guard=SubjectGuards({'front':front},side_veto=False)
    ids=np.load(a.repair_dir/'local_registration_audit_20260911/paired_points.npz')['耳前_pixel_ids']
    toward=left.inverse[:3,2].copy();toward/=np.linalg.norm(toward)
    base=left.head_points[ids]+.0008*toward
    if not left.head_hit.ravel()[ids].all():raise ValueError('固定耳前射线存在未命中')
    # 左视横纵像素格 + 视线深度格，映射到三维；保持真实欧氏半径限制。
    lateral=np.array([(x,y) for y in range(-8,9) for x in range(-8,9) if x*x+y*y<=64])
    vectors=lateral@((left.inverse[:3,:2]*2/511).T)
    offsets=(vectors[:,None,:]+np.arange(-20,21)[None,:,None]*.001*toward).reshape(-1,3)
    pixel_shift=np.repeat(np.linalg.norm(lateral,axis=1),41)
    distance=np.linalg.norm(offsets,axis=1)
    keep=distance<=.02000001;offsets,pixel_shift,distance=offsets[keep],pixel_shift[keep],distance[keep]
    order=np.argsort(distance,kind='stable');offsets,pixel_shift,distance=offsets[order],pixel_shift[order],distance[order]
    limits=[0,1,2,4,8]
    best=np.full((len(ids),len(limits),3),np.nan);cost=np.full((len(ids),len(limits)),np.inf)
    def legal(points):
        ok=guard.points(points,collision=False)==0
        signed=front.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)),nsamples=3).numpy()
        pix,_,valid=left.pixels(points)
        hair=np.zeros(len(points),bool);hair[valid]=left.hair_domain[pix[valid,1],pix[valid,0]]
        rays=np.c_[points+2*toward,np.broadcast_to(-toward,points.shape)]
        visible=left.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()>=2-.00001
        return ok&(signed>=.0004)&hair&visible
    for start in range(0,len(ids),16):
        roots=base[start:start+16]
        points=(roots[:,None]+offsets).reshape(-1,3)
        ok=legal(points).reshape(len(roots),len(offsets))
        for j,limit in enumerate(limits):
            valid=ok&(pixel_shift<=limit+1e-8)
            found=valid.any(1);ix=valid.argmax(1);rows=np.flatnonzero(found)
            best[start+rows,j]=roots[rows]+offsets[ix[rows]];cost[start+rows,j]=distance[ix[rows]]
        if start%160==0:print(f'searched {min(start+16,len(ids))}/{len(ids)}',flush=True)
    report={'points':len(ids),'candidate_offsets_per_point':len(offsets),'grid':{'lateral_step_px':1,'depth_step_mm':1,'max_radius_mm':20},'budgets':{},'continuity':{}}
    for j,limit in enumerate(limits):
        report['budgets'][str(limit)]={str(mm):int((cost[:,j]<=mm/1000+1e-8).sum()) for mm in [2,5,10,20]}
    lookup={int(v):i for i,v in enumerate(ids)}
    edges=[]
    for i,v in enumerate(ids):
        for neighbor in [int(v)+1,int(v)+512]:
            if neighbor in lookup and (neighbor//512==v//512 or neighbor==v+512):edges.append((i,lookup[neighbor]))
    edges=np.asarray(edges)
    for limit in [2,8]:
        j=limits.index(limit);found=np.isfinite(cost[:,j]);active=edges[found[edges].all(1)]
        failures=0;lengths=[];jumps=[]
        for i,k in active:
            q,r=best[i,j],best[k,j];length=np.linalg.norm(q-r);lengths.append(length)
            jumps.append(np.linalg.norm((q-base[i])-(r-base[k])))
            samples=np.linspace(q,r,max(2,int(np.ceil(length/.00025))+1))
            if not legal(samples).all():failures+=1
        report['continuity'][str(limit)]={'found_points':int(found.sum()),'neighbor_pairs_total':len(edges),'pairs_with_both_endpoints':len(active),'illegal_straight_connections':failures,
            'displacement_jump_mm_50_90_max':(np.quantile(jumps,[.5,.9,1])*1000).tolist() if jumps else None,
            'connection_length_mm_50_90_max':(np.quantile(lengths,[.5,.9,1])*1000).tolist() if lengths else None}
        pixels,_,valid=left.pixels(best[found,j])
        covered=np.zeros((512,512),np.uint8)
        covered[pixels[valid,1],pixels[valid,0]]=1
        report['continuity'][str(limit)]['distinct_endpoint_pixels']=int(covered.sum())
        report['continuity'][str(limit)]['original_roi_pixels_covered_by_endpoints']=int(covered.ravel()[ids].sum())
        inflated=cv2.dilate(covered,np.ones((3,3),np.uint8))
        report['continuity'][str(limit)]['original_roi_pixels_within_1px_of_endpoints']=int(inflated.ravel()[ids].sum())
    photo=Image.open(a.data_dir/'raw_img.png').convert('RGB').resize((512,512))
    side=Image.open(a.data_dir/'flux_redrawn/left.png').convert('RGB').resize((512,512))
    canvas=Image.new('RGB',(1536,585),'white');pen=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',20)
    for col,limit in enumerate([0,2,8]):
        j=limits.index(limit);im=side.copy();draw=ImageDraw.Draw(im)
        for i,v in enumerate(ids):
            color=(0,220,150) if cost[i,j]<=.005+1e-8 else ((255,195,0) if np.isfinite(cost[i,j]) else (255,45,65))
            draw.point((int(v%512),int(v//512)),fill=color)
        canvas.paste(im,(512*col,35));pen.text((512*col+8,8),f'侧图允许偏移 {limit}px；半径≤20mm',font=font,fill='black')
    pen.text((8,555),'绿：5mm内找到；黄：5–20mm找到；红：采样未找到。点可行不等于能形成连续表面。',font=font,fill='black')
    canvas.save(a.output_dir/'neighborhood_comparison.png')
    np.savez_compressed(a.output_dir/'solutions.npz',pixel_ids=ids,base=base,best=best,cost_m=cost,pixel_limits=limits)
    report['limits']='固定原图/头模，使用原图约束配准。有限格点搜索不是连续不可行证明；邻边仅检查最近解的直线连接，未求平滑曲面，也未验证三角形内部。合法点还要求侧图hair seg及精确左视可见性。端点覆盖不是发丝渲染覆盖。'
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
