"""仅调整 front 正交投影的像素尺度与中心，检查标定不一致影响。"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 h=np.load(a.data_dir/'pde_governance/volume_partition_integration/subject_repair_v32/geometry/template_head/template_head.npz');v=h['vertices'];cam=load_observation_camera(a.data_dir,'front')
 q=np.c_[v,np.ones(len(v))]@cam.T;q=q/q[:,3:];uv=(q[:,:2]+1)*255.5;old_center=uv.mean(0);old_size=uv.max(0)-uv.min(0)
 side_cam=load_observation_camera(a.data_dir,'left');s=np.c_[v,np.ones(len(v))]@side_cam.T;s=s/s[:,3:];suv=(s[:,:2]+1)*255.5;side_size=suv.max(0)-suv.min(0);ratio=float(np.mean(side_size/old_size));target_center=np.array([256.,246.])
 clip_center=np.mean(q[:,:2],0);target_clip=(target_center/255.5-1);A=np.eye(4);A[0,0]=ratio;A[1,1]=ratio;A[0,3]=target_clip[0]-ratio*clip_center[0];A[1,3]=target_clip[1]-ratio*clip_center[1];new=A@cam
 nq=np.c_[v,np.ones(len(v))]@new.T;nq=nq/nq[:,3:];nuv=(nq[:,:2]+1)*255.5
 report={'old_bbox':np.r_[uv.min(0),uv.max(0)].tolist(),'old_size':old_size.tolist(),'side_size':side_size.tolist(),'scale_ratio':ratio,'new_bbox':np.r_[nuv.min(0),nuv.max(0)].tolist(),'new_size':(nuv.max(0)-nuv.min(0)).tolist(),'new_center':nuv.mean(0).tolist(),'transform':'clip_xy_scale_and_center_only'}
 # 保留现有 front 关键点，报告相机变化对其投影的影响；不把自动关键点当作新真值。
 try:
  lm=np.load(a.data_dir/'pde_governance/volume_partition_integration/subject_repair_v29/front_pose_holdout/landmarks.npz');src=lm['source'];target=lm['target'];
  old_l=np.c_[src,np.ones(len(src))]@cam.T;old_l=old_l[:,:2]/old_l[:,3:];old_l=(old_l+1)*255.5;new_l=np.c_[src,np.ones(len(src))]@new.T;new_l=new_l[:,:2]/new_l[:,3:];new_l=(new_l+1)*255.5
  report['landmark_error_px']={'old_mean':float(np.linalg.norm(old_l-target,axis=1).mean()),'new_mean':float(np.linalg.norm(new_l-target,axis=1).mean()),'old_holdout':float(np.linalg.norm(old_l[lm['holdout']]-target[lm['holdout']],axis=1).mean()),'new_holdout':float(np.linalg.norm(new_l[lm['holdout']]-target[lm['holdout']],axis=1).mean())}
 except Exception as e: report['landmark_error_unavailable']=str(e)
 np.savez(a.output_dir/'front_camera_scaled.npz',camera=new,scale_ratio=ratio,source_camera=cam)
 (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
