"""修正 ICP 后按左图人工诊断区域分析深度与原图语义残差。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--repair-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),o3d.utility.Vector3iVector(d['faces']))
    if not head.is_watertight():
        raise ValueError('signed distance requires watertight head')
    glb = o3d.io.read_triangle_mesh(str(a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'))
    old = np.load(a.data_dir/'pixal3d/glb_to_world.npz')['icp_T']
    fixed = np.load(a.repair_dir/'icp_order_audit_20260911/transforms.npz')['consistent_indices']
    delta = fixed@np.linalg.inv(old)
    glb.transform(delta)
    cam = load_observation_camera(a.data_dir,'left')@np.linalg.inv(delta)
    front_cam = load_observation_camera(a.data_dir,'front')
    seg = {v: cv2.imread(str(a.data_dir/'maps/seg'/f'{v}.png'),0)>127 for v in ['front','left']}
    def chart(camera,mask,mesh):
        return VisibleSurfaceGrowth(camera,mask.astype(int),np.zeros((512,512,2)),mesh)
    hc,gc = chart(cam,seg['left'],head),chart(cam,seg['left'],glb)
    front = chart(front_cam,seg['front'],head)
    toward = hc.inverse[:3,2].copy(); toward /= np.linalg.norm(toward)
    ft = front.inverse[:3,2].copy(); ft /= np.linalg.norm(ft)
    gap = np.load(a.repair_dir/'registration_counterfactual_20260911/evidence.npz')
    mask = np.zeros(512*512,bool); mask[gap['pixel_indices']]=True
    feasible = np.zeros_like(mask); feasible[gap['pixel_indices']]=gap['fixed_indices_feasible'].any(1)
    # 显式人工矩形仅用于分区诊断，不作为解剖关键点或优化监督。
    rois = {'额角':(140,120,185,165),'太阳穴':(185,125,230,167),'耳前':(195,167,238,207)}
    yy,xx = np.indices((512,512)); xx,yy=xx.ravel(),yy.ravel()
    dist = cv2.distanceTransform((~seg['front']).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    photo = Image.open(a.data_dir/'raw_img.png').convert('RGB').resize((512,512))
    side = Image.open(a.data_dir/'flux_redrawn/left.png').convert('RGB').resize((512,512))
    canvas = Image.new('RGB',(1536,3*570+55),'white')
    draw=ImageDraw.Draw(canvas); font=ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',20)
    report={}; arrays={}
    def measure(points,scene):
        rays=np.c_[points+2*ft,np.broadcast_to(-ft,points.shape)]
        visible=scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()>=2-.0001
        uv,_=front.project(points); ids=np.rint(uv).astype(int)
        valid=((ids>=0)&(ids<512)).all(1)
        positive=np.zeros(len(points),bool); outside=np.full(len(points),np.nan)
        positive[valid]=seg['front'][ids[valid,1],ids[valid,0]]
        outside[valid]=dist[ids[valid,1],ids[valid,0]]
        bad=valid&visible&~positive
        values=outside[bad]
        return {'visible_nonhair':int(bad.sum()),'occluded':int((~visible&valid).sum()),
            'visible_hair':int((visible&positive).sum()),'out_of_image':int((~valid).sum()),
            'nonhair_distance_to_hair_px_50_90':np.quantile(values,[.5,.9]).tolist() if len(values) else None},uv,bad,visible
    for row,(name,(x0,y0,x1,y1)) in enumerate(rois.items()):
        keep=mask&(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)&hc.head_hit.ravel()&gc.head_hit.ravel()
        ids=np.flatnonzero(keep)
        gp=gc.head_points[ids]; hp=hc.head_points[ids]+.0008*toward
        gstat,guv,gbad,gvis=measure(gp,gc.scene)
        hstat,huv,hbad,hvis=measure(hp,hc.scene)
        signed=hc.scene.compute_signed_distance(o3d.core.Tensor(gp.astype(np.float32)),nsamples=3).numpy()
        # 同一左视射线的首交点差，不声称为同一解剖点。
        ray_depth=(hc.head_points[ids]-gp)@toward*1000
        shift=huv-guv
        report[name]={'roi_xyxy':list(rois[name]),'paired_gap_pixels':len(ids),'sampled_feasible':int(feasible[ids].sum()),
            'glb_self_visibility':gstat,'head_outside_0_8mm':hstat,
            'glb_inside_head':int((signed<0).sum()),
            'glb_signed_distance_mm_10_50_90':np.quantile(signed*1000,[.1,.5,.9]).tolist(),
            'head_in_front_of_glb_ray_mm_10_50_90':np.quantile(ray_depth,[.1,.5,.9]).tolist(),
            'paired_projection_distance_px_50_90':np.quantile(np.linalg.norm(shift,axis=1),[.5,.9]).tolist(),
            'paired_projection_shift_xy_median_px':np.median(shift,axis=0).tolist()}
        arrays[name+'_pixel_ids']=ids; arrays[name+'_glb_points']=gp; arrays[name+'_head_offset_points']=hp
        images=[side.copy(),photo.copy(),photo.copy()]
        sd=ImageDraw.Draw(images[0]); sd.rectangle(rois[name],outline=(255,210,0),width=2)
        for x,y in zip(xx[ids][::3],yy[ids][::3]):sd.point((int(x),int(y)),fill=(255,70,60))
        for im,uv,bad,vis in [(images[1],guv,gbad,gvis),(images[2],huv,hbad,hvis)]:
            pen=ImageDraw.Draw(im)
            for i in range(0,len(uv),3):
                x,y=uv[i]
                if 0<=x<512 and 0<=y<512:
                    color=(255,45,65) if bad[i] else ((0,210,140) if vis[i] else (0,150,255))
                    pen.ellipse((x-1,y-1,x+1,y+1),fill=color)
        titles=[f'{name}：固定左图诊断框，{len(ids)} 点',f'GLB 自遮挡：可见非头发 {gbad.sum()}',f'头模外侧 0.8mm：可见非头发 {hbad.sum()}']
        for col,im in enumerate(images):
            draw.text((col*512+6,row*570+5),titles[col],font=font,fill='black')
            canvas.paste(im,(col*512,row*570+35))
        print(name,report[name],flush=True)
    draw.text((8,1712),'红：原图可见非头发；绿：原图可见头发；蓝：自身几何遮挡。框为人工诊断区域，不是解剖对应真值。',font=font,fill='black')
    report['limits']='仅当前个体与修正ICP。局部框是人工图像区域，首交点不是解剖同名点；头模外侧0.8mm不是失败发丝。原图seg语义距离不是解剖重投影误差。'
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(a.output_dir/'paired_points.npz',**arrays)
    canvas.save(a.output_dir/'local_residuals.png')


if __name__=='__main__':
    main()
