"""同一耳前 GLB 表面点的配准消融，以及生成前后局部图像对照。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--repair-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    old=np.load(a.data_dir/'pixal3d/glb_to_world.npz')['icp_T']
    fixed=np.load(a.repair_dir/'icp_order_audit_20260911/transforms.npz')['consistent_indices']
    delta=fixed@np.linalg.inv(old)
    points=np.load(a.repair_dir/'local_registration_audit_20260911/paired_points.npz')['耳前_glb_points']
    inv=np.linalg.inv(delta)
    saved_points=points@inv[:3,:3].T+inv[:3,3]
    glb=o3d.io.read_triangle_mesh(str(a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(glb))
    cam=load_observation_camera(a.data_dir,'front')
    toward=np.linalg.inv(cam)[:3,2];toward/=np.linalg.norm(toward)
    seg=cv2.imread(str(a.data_dir/'maps/seg/front.png'),0)>127
    dist=cv2.distanceTransform((~seg).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    canvas=Image.new('RGB',(1536,960),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/TTF/LXGWWenKai-Regular.ttf',21)
    photo=Image.open(a.data_dir/'raw_img.png').convert('RGB').resize((512,512))
    report={'fixed_glb_surface_points':len(points),'variants':{}}
    variants={'仅 Umeyama':np.linalg.inv(old),'原有 ICP':np.eye(4),'修正 ICP':delta}
    for col,(name,t) in enumerate(variants.items()):
        gp=saved_points@t[:3,:3].T+t[:3,3]
        ti=np.linalg.inv(t)
        origins=(gp+2*toward)@ti[:3,:3].T+ti[:3,3]
        direction=-toward@ti[:3,:3].T
        rays=np.c_[origins,np.broadcast_to(direction,origins.shape)]
        visible=scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()>=2-.0001
        clip=np.c_[gp,np.ones(len(gp))]@cam.T
        uv=(clip[:,:2]/clip[:,3:]+1)*255.5
        ids=np.rint(uv).astype(int);valid=((ids>=0)&(ids<512)).all(1)
        positive=np.zeros(len(ids),bool);distance=np.full(len(ids),np.nan)
        positive[valid]=seg[ids[valid,1],ids[valid,0]]
        distance[valid]=dist[ids[valid,1],ids[valid,0]]
        bad=visible&valid&~positive
        report['variants'][name]={'visible_nonhair':int(bad.sum()),'visible_hair':int((visible&positive).sum()),
            'occluded':int((~visible&valid).sum()),'out_of_image':int((~valid).sum()),
            'visible_nonhair_distance_px_50_90':np.quantile(distance[bad],[.5,.9]).tolist() if bad.any() else None,
            'projection_xy_median':np.median(uv,axis=0).tolist()}
        im=photo.copy();pen=ImageDraw.Draw(im)
        for i,(x,y) in enumerate(uv):
            if valid[i]:
                color=(255,40,60) if bad[i] else ((0,200,120) if visible[i] else (0,150,255))
                pen.point((int(x),int(y)),fill=color)
        draw.text((512*col+8,8),f'{name}：可见非头发 {bad.sum()}/{len(points)}',font=font,fill='black')
        canvas.paste(im,(512*col,40))
    box=(180,140,275,225)
    for col,(path,title) in enumerate([(a.data_dir/'blender_renders/left.png','生成前：GLB 渲染'),(a.data_dir/'flux_redrawn/left.png','生成后：FLUX'),(a.data_dir/'maps/seg/left.png','现有侧图头发 seg')]):
        im=Image.open(path).convert('RGB').resize((512,512))
        pen=ImageDraw.Draw(im);pen.rectangle((195,167,238,207),outline=(255,200,0),width=1)
        crop=im.crop(box).resize((380,340))
        canvas.paste(crop,(512*col+60,600))
        draw.text((512*col+8,565),title,font=font,fill='black')
    report['limits']='同一616个GLB表面点；原图seg与标定固定。生成前后局部图仅作视觉核查，没有可靠的生成前头发seg，不报告FLUX新增头发面积。仅Umeyama仍是关键点配准，不代表原生GLB完全无误。'
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    canvas.save(a.output_dir/'ear_source_comparison.png')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
