"""验证 GLB 头发壳层与头模距离及 front/left 投影覆盖。"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--repair-dir',type=Path,required=True);p.add_argument('--shell-dir',type=Path,required=True);a=p.parse_args()
 if not a.shell_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出必须按Image ID聚合')
 shell=trimesh.load(a.shell_dir/'hair_outer_surface.obj',process=False)
 h=np.load(a.repair_dir/'geometry/template_head/template_head.npz');hv=h['vertices']
 tree=cKDTree(hv);dist,_=tree.query(np.asarray(shell.vertices),k=1)
 report={'shell_vertices':len(shell.vertices),'shell_faces':len(shell.faces),'head_nearest_distance_mm_q10_q50_q90':(np.quantile(dist,[.1,.5,.9])*1000).tolist(),'shell_vertices_inside_head_proxy':int((dist<.0004).sum())}
 seg={v:cv2.imread(str(a.data_dir/'maps/seg'/f'{v}.png'),0)>127 for v in ['front','left']}
 for view in ['front','left']:
  cam=load_observation_camera(a.data_dir,view);q=np.c_[shell.vertices,np.ones(len(shell.vertices))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ids=np.rint(uv).astype(int);valid=((ids>=0)&(ids<512)).all(1);pos=np.zeros(len(uv),bool);pos[valid]=seg[view][ids[valid,1],ids[valid,0]]
  roi=(ids[:,0]>=195)&(ids[:,0]<238)&(ids[:,1]>=167)&(ids[:,1]<207) if view=='left' else (ids[:,0]>=150)&(ids[:,0]<240)&(ids[:,1]>=120)&(ids[:,1]<210)
  report[view]={'projected_valid':int(valid.sum()),'projected_hair':int((valid&pos).sum()),'projected_nonhair':int((valid&~pos).sum()),'roi_shell_pixels':int((valid&roi).sum()),'roi_hair_pixels':int((valid&roi&pos).sum())}
  image=cv2.imread(str(a.data_dir/('raw_img.png' if view=='front' else 'flux_redrawn/left.png')));image=cv2.resize(image,(512,512));
  for x,y,ok in zip(ids[::20,0],ids[::20,1],(valid&pos)[::20]):
   if 0<=x<512 and 0<=y<512:cv2.circle(image,(x,y),1,(0,255,0) if ok else (0,0,255),-1)
  cv2.imwrite(str(a.shell_dir/f'{view}_projection.png'),image)
 report['limits']='头模距离使用顶点KD树，仅是保守近似，不是点到三角面有符号距离；壳层来自旧配准GLB，未应用原图约束配准；投影统计使用原始front seg和现有left seg。'
 (a.shell_dir/'projection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
