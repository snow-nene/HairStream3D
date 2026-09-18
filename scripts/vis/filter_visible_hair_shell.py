"""按 front/left 首交点与 hair seg 筛选 GLB 头发壳层三角面。"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera
from lib.multiview_hair_evidence import classify_evidence, EvidenceState

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--shell-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出必须按Image ID聚合')
 a.output_dir.mkdir(parents=True,exist_ok=True)
 src=o3d.io.read_triangle_mesh(str(a.shell_dir/'hair_outer_surface.obj'));v=np.asarray(src.vertices);f=np.asarray(src.triangles);c=v[f].mean(1)
 hv=np.asarray(np.load(a.data_dir/'pde_governance/volume_partition_integration/subject_repair_v32/geometry/template_head/template_head.npz')['vertices']);tree=cKDTree(hv)
 scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(src))
 keep=np.zeros(len(f),bool);front_bad=np.zeros(len(f),bool);stats={}
 for view in ['front','left']:
  cam=load_observation_camera(a.data_dir,view);q=np.c_[c,np.ones(len(c))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ids=np.rint(uv).astype(int);valid=((ids>=0)&(ids<512)).all(1);seg=cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'),0)>127;hair=np.zeros(len(c),bool);hair[valid]=seg[ids[valid,1],ids[valid,0]]
  # 从点沿相机方向反向投射；若首交点距离接近0，则该面是自身可见面。
  toward=np.linalg.inv(cam)[:3,2];toward/=np.linalg.norm(toward);rays=np.c_[c+2*toward,np.broadcast_to(-toward,c.shape)]
  hit=scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy();visible=hit>=1.999
  # 面级筛选：仅中心通过会让三角形边缘越过 front 发际线。
  samples=np.concatenate([v[f], .5*(v[f][:, [0,1,2]]+v[f][:, [1,2,0]]), c[:,None]],axis=1).reshape(-1,3)
  sq=np.c_[samples,np.ones(len(samples))]@cam.T; suv=(sq[:,:2]/sq[:,3:]+1)*255.5;si=np.rint(suv).astype(int);sv=((si>=0)&(si<512)).all(1);sh=np.zeros(len(samples),bool);sh[sv]=seg[si[sv,1],si[sv,0]]
  sr=scene.cast_rays(o3d.core.Tensor(np.c_[samples+2*toward,np.broadcast_to(-toward,samples.shape)].astype(np.float32)))['t_hit'].numpy()>=1.999
  sample_ok=(sv&sh&sr).reshape(len(f),7)
  if view=='front':
   # front 不可见的面由头模遮挡，不能因缺少 front 语义而删除。
   visible_nonhair=(sv&sr&~sh).reshape(len(f),7).any(1)
   front_bad |= visible_nonhair
   accepted=(sample_ok & ~visible_nonhair[:,None]).all(1)
  else:
   accepted=sample_ok.all(1)
  keep|=accepted
  stats[view]={'faces':len(f),'valid':int(valid.sum()),'visible':int(visible.sum()),'hair':int((valid&hair).sum()),'accepted':int(accepted.sum())}
 keep &= ~front_bad
 used,remap=np.unique(f[keep],return_inverse=True);out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v[used]),o3d.utility.Vector3iVector(remap.reshape(-1,3)));o3d.io.write_triangle_mesh(str(a.output_dir/'visible_hair_shell.obj'),out)
 # 投影筛选后的顶点，报告耳前覆盖和头模距离。
 # 统一证据状态统计（面中心）；最终保留仍使用上面的七点面级硬检查。
 front_cam=load_observation_camera(a.data_dir,'front');left_cam=load_observation_camera(a.data_dir,'left')
 def view_flags(cam,mask):
  q=np.c_[c,np.ones(len(c))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ii=np.rint(uv).astype(int);ok=((ii>=0)&(ii<512)).all(1);hair=np.zeros(len(c),bool);hair[ok]=mask[ii[ok,1],ii[ok,0]]
  toward=np.linalg.inv(cam)[:3,2];toward/=np.linalg.norm(toward);hit=scene.cast_rays(o3d.core.Tensor(np.c_[c+2*toward,np.broadcast_to(-toward,c.shape)].astype(np.float32)))['t_hit'].numpy();vis=hit>=1.999
  return vis,hair
 fv,fh=view_flags(front_cam,cv2.imread(str(a.data_dir/'maps/seg/front.png'),0)>127);lv,lh=view_flags(left_cam,cv2.imread(str(a.data_dir/'maps/seg/left.png'),0)>127)
 clearance=tree.query(c)[0]
 state,score,accepted=classify_evidence(front_visible=fv,front_hair=fh,side_support=lv&lh,head_clearance=clearance)
 report={'source_faces':len(f),'front_veto_faces':int(front_bad.sum()),'selected_faces':int(keep.sum()),'selected_vertices':len(used),'stats':stats,'evidence_state_counts':{name:int((state==value).sum()) for name,value in EvidenceState.__members__.items()},'selected_state_counts':{name:int((state[keep]==value).sum()) for name,value in EvidenceState.__members__.items()}}
 dist=tree.query(v[used])[0];report['head_distance_mm_q10_q50_q90']=(np.quantile(dist,[.1,.5,.9])*1000).tolist()
 for view in ['front','left']:
  cam=load_observation_camera(a.data_dir,view);q=np.c_[v[used],np.ones(len(used))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ids=np.rint(uv).astype(int);valid=((ids>=0)&(ids<512)).all(1);seg=cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'),0)>127;hair=np.zeros(len(used),bool);hair[valid]=seg[ids[valid,1],ids[valid,0]];roi=(ids[:,0]>=195)&(ids[:,0]<238)&(ids[:,1]>=167)&(ids[:,1]<207) if view=='left' else np.zeros(len(used),bool);report[view+'_selected_vertices']={'valid':int(valid.sum()),'hair':int((valid&hair).sum()),'nonhair':int((valid&~hair).sum()),'ear_roi':int((valid&roi).sum()),'ear_roi_hair':int((valid&roi&hair).sum())}
 (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
