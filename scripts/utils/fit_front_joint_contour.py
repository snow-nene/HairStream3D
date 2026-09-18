#!/usr/bin/env python3
"""轮廓带+关键点的 front 头模联合校准候选。"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np, open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.utils.align_glb_lmk import get_lmk
from scripts.utils.fit_front_pose_so3 import nearest_rotation,project_landmarks,build_param
from scripts.recon_3d.recon3D import load_calib

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--img-id',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
 base=ROOT/'results/multiview_data'/args.img_id; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 mask=cv2.imread(str(out/'target_head_mask.png'),0)>127; contour=cv2.imread(str(out/'target_contour_band.png'),0)>127
 pts2=np.argwhere(contour)[:,[1,0]].astype(float); pts2=pts2[::max(1,len(pts2)//2500)]
 old=np.load(base/'maps/param/front.npy',allow_pickle=True).item(); ortho=float(old.get('ortho_ratio',.2)); R0=nearest_rotation(old['R']); c0=np.asarray(old['center']).reshape(3); s0=float(np.asarray(old['scale']).reshape(-1)[0])
 mesh_npz=np.load(base/'pde_governance/volume_partition_integration/subject_repair_v32/geometry/template_head/template_head.npz'); verts=mesh_npz['vertices'][::15]
 image=cv2.imread(str(base/'raw_img.png')); lmk=get_lmk(cv2.resize(image,(512,512)))[:68,:2].astype(float)
 src=np.load(ROOT/'data/landmark_id_uschair.obj.npy') if False else None
 # landmark source from canonical head model
 from lib.mesh_util import load_obj_mesh
 from lib.util.opt_lmk import load_point_ids
 hv,_=load_obj_mesh(str(ROOT/'data/head_model.obj')); ids=load_point_ids(str(ROOT/'data/landmark_id_uschair.obj')); source=np.asarray(hv[ids[:68]],float)
 holdout=np.arange(len(source))%3==0
 def unpack(x):
  R=Rotation.from_rotvec(x[:3]).as_matrix(); c=np.array([x[3],x[4],c0[2]]); return R,c,np.exp(x[5])
 def proj(p,R,c,s): return project_landmarks(p,R,c,s,ortho,512)
 def residual(x):
  R,c,s=unpack(x); pp=proj(verts,R,c,s); tree=cKDTree(pp); d=tree.query(pts2,k=1)[0]; pred=proj(source,R,c,s); ld=(pred-lmk)[~holdout].ravel(); return np.r_[d,0.35*ld]
 x0=np.r_[Rotation.from_matrix(R0).as_rotvec(),c0[:2],np.log(s0)]
 res=least_squares(residual,x0,loss='soft_l1',f_scale=3,max_nfev=250,verbose=0)
 R,c,s=unpack(res.x); param=build_param(R,c,s,ortho); np.save(out/'front_joint.npy',param); calib=load_calib(str(out/'front_joint.npy'),loadSize=1024).numpy()
 oldp=proj(source,old['R'],c0,s0); newp=proj(source,R,c,s); olde=np.linalg.norm(oldp-lmk,axis=1); newe=np.linalg.norm(newp-lmk,axis=1)
 np.savez_compressed(out/'joint_metrics.npz',target_contour=pts2,old_landmarks=oldp,new_landmarks=newp,observed_landmarks=lmk)
 report={'status':'candidate_only_not_installed','vertices_sampled':int(len(verts)),'target_contour_samples':int(len(pts2)),'legacy_landmark_mean_px':float(olde.mean()),'joint_landmark_mean_px':float(newe.mean()),'legacy_landmark_holdout_px':float(olde[::3].mean()),'joint_landmark_holdout_px':float(newe[::3].mean()),'scale':float(s),'center':c.tolist(),'det_R':float(np.linalg.det(R))}
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
