"""诊断分支：固定原图相机，以原图关键点约束 GLB 刚性配准。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.utils.align_glb_lmk import get_lmk
from scripts.utils.fit_front_pose_dense import render_pixels_to_gltf_surface
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera
from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--repair-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    a.data_dir=a.data_dir.resolve();a.repair_dir=a.repair_dir.resolve();a.output_dir=a.output_dir.resolve()
    if not a.output_dir.is_relative_to(a.data_dir):raise ValueError('输出须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    raw=o3d.io.read_triangle_mesh(str(next((a.data_dir/'pixal3d').glob('*.glb'))))
    cache=a.output_dir/'detected_landmarks.npz'
    if cache.exists():
        detected=np.load(cache);render_lm=detected['render'];target=detected['photo']
    else:
        render_lm=get_lmk(cv2.imread(str(a.data_dir/'pixal3d/render_front.png')))
        target=get_lmk(cv2.imread(str(a.data_dir/'raw_img.png')))
        if render_lm is None or target is None:raise ValueError('关键点检测失败')
        np.savez(cache,render=render_lm,photo=target)
    raw_lm,valid=render_pixels_to_gltf_surface(render_lm[:,:2],np.load(a.data_dir/'pixal3d/render_camera.npz'),raw)
    saved=np.load(a.data_dir/'pixal3d/glb_to_world.npz')
    u=np.eye(4);u[:3,:3]=float(saved['umeyama_scale'].ravel()[0])*saved['umeyama_R'];u[:3,3]=saved['umeyama_t']
    pts=raw_lm@u[:3,:3].T+u[:3,3]
    cam=load_observation_camera(a.data_dir,'front')
    def project(points):
        q=np.c_[points,np.ones(len(points))]@cam.T
        return (q[:,:2]/q[:,3:]+1)*255.5
    holdout=(np.arange(68)%3==0)&valid;train=valid&~holdout
    v,f=load_obj_mesh('data/head_model.obj')
    head=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f));head.compute_vertex_normals()
    lm=v[load_point_ids('data/landmark_id_uschair.obj')]
    lo,hi=lm.min(0)-.05,lm.max(0)+.05
    selected=((v>=lo)&(v<=hi)).all(1)
    tree=cKDTree(v[selected]);nv=np.asarray(head.vertex_normals)[selected];tv=v[selected]
    rv=np.asarray(raw.vertices)@u[:3,:3].T+u[:3,3]
    surface=rv[((rv>=lo)&(rv<=hi)).all(1)][::30]
    pivot=pts[train].mean(0)
    def matrix(x):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();t[:3,3]=pivot+x[3:]-t[:3,:3]@pivot
        return t
    def move(points,t):return points@t[:3,:3].T+t[:3,3]
    def residual(x,weight):
        t=matrix(x);q=move(surface,t)
        _,ids=tree.query(q)
        geometric=((q-tv[ids])*nv[ids]).sum(1)/.01/np.sqrt(len(q))
        photo=(project(move(pts[train],t))-target[train,:2]).ravel()/3/np.sqrt(train.sum()*2)
        return np.r_[geometric,weight*photo,x[:3]/.35*.05,x[3:]/.05*.05]
    fixed=np.load(a.repair_dir/'icp_order_audit_20260911/transforms.npz')['consistent_indices']
    variants={'umeyama':np.eye(4),'saved_icp':saved['icp_T'],'fixed_icp':fixed}
    report={'training_landmarks':int(train.sum()),'holdout_landmarks':int(holdout.sum()),'valid_lifted_landmarks':int(valid.sum()),
            'geometry_samples':len(surface),'preset_candidate':'photo_weight_1','variants':{}}
    for w in [0,.25,1,4]:
        fit=least_squares(residual,np.zeros(6),args=(w,),bounds=([-.5]*3+[-.05]*3,[.5]*3+[.05]*3),max_nfev=200)
        name=f'photo_weight_{w:g}';variants[name]=matrix(fit.x)
    photo=Image.open(a.data_dir/'raw_img.png').convert('RGB')
    board=Image.new('RGB',(512*len(variants),550),'white');pen=ImageDraw.Draw(board)
    for col,(name,t) in enumerate(variants.items()):
        pred=project(move(pts,t));errors=np.linalg.norm(pred-target[:,:2],axis=1)
        q=move(surface,t);_,ids=tree.query(q);geom=((q-tv[ids])*nv[ids]).sum(1)*1000
        report['variants'][name]={'train_mean_px':float(errors[train].mean()),'holdout_mean_px':float(errors[holdout].mean()),
            'holdout_q90_px':float(np.quantile(errors[holdout],.9)),'geometry_plane_rms_mm':float(np.sqrt(np.mean(geom**2)))}
        im=photo.copy();d=ImageDraw.Draw(im)
        for i in np.flatnonzero(valid):
            x,y=pred[i];tx,ty=target[i,:2];color=(255,50,50) if holdout[i] else (0,210,180)
            d.line((tx,ty,x,y),fill=color,width=1);d.ellipse((x-2,y-2,x+2,y+2),outline=color)
        board.paste(im,(col*512,35));pen.text((col*512+8,8),name,fill='black')
    np.savez(a.output_dir/'transforms.npz',umeyama_transform=u,**variants)
    np.savez(a.output_dir/'landmarks.npz',source=pts,target=target[:,:2],train=train,holdout=holdout)
    report['limits']='原图为唯一目标；GLB渲染仅用于自动关键点的真实表面定位。自动关键点非人工真值。权重1事先指定，其他权重仅敏感性对照；留出点未参与残差或候选选择。几何项采用固定稀疏采样，与旧密集ICP数值不可直接比较。'
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));board.save(a.output_dir/'landmark_comparison.png')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
