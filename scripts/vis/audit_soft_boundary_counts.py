"""统计 front 软边界对现有发丝和耳前候选的影响，不执行重建。"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.ndimage import distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth,load_observation_camera,sample_polyline

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--repair-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):raise ValueError('输出必须按Image ID聚合')
 a.output_dir.mkdir(parents=True,exist_ok=True)
 d=np.load(a.repair_dir/'geometry/template_head/template_head.npz');mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),o3d.utility.Vector3iVector(d['faces']))
 cams={};seg={};charts={}
 for v in ['front','left','right','back']:
  seg[v]=cv2.imread(str(a.data_dir/'maps/seg'/f'{v}.png'),0)>127;charts[v]=VisibleSurfaceGrowth(load_observation_camera(a.data_dir,v),seg[v].astype(int),np.zeros((512,512,2)),mesh)
 front=charts['front'];fd=distance_transform_edt(~front.hair_domain)
 strands=np.load(a.repair_dir/'full_exact_2mm_20260911/all_root_prefixes.npz')['strands']
 # 稀疏采样整条基线，保留连续段统计；这是约束敏感度，不是最终重建接受率。
 points=np.concatenate([sample_polyline(s,.001) for s in strands[:11189]])
 e=np.load(a.repair_dir/'local_registration_audit_20260911/paired_points.npz');ear=e['耳前_glb_points']
 report={'baseline_strands':11189,'baseline_samples':len(points),'bands':{},'ear_points':len(ear)}
 def classify(points,band):
  reason=np.zeros(len(points),np.int8);ids,depth,valid=front.pixels(points);ii=np.flatnonzero(valid);x,y=ids[valid].T
  vis=np.zeros(len(points),bool);vis[ii]=~front.head_hit[y,x]|(depth[valid]>=front.head_z[y,x]-1e-5)
  positive=np.zeros(len(points),bool);positive[ii]=front.hair_domain[y,x]
  outside=vis&~positive;near=np.zeros(len(points),bool);near[ii]=fd[y,x]<=band
  reason[outside&~near]=4;reason[~valid]=5
  # 独立碰撞距离；仅作冲突类别统计。
  nearp=front.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
  clearance=((points-nearp['points'].numpy())*nearp['primitive_normals'].numpy()).sum(1)
  reason[(reason==0)&(clearance<.00045)]=3
  return reason
 for band in [0,2,4,8]:
  r=classify(points,band);er=classify(ear,band)
  report['bands'][str(band)]={'baseline_ok':int((r==0).sum()),'front_visible_nonhair_hard':int((r==4).sum()),'out_of_image':int((r==5).sum()),'collision':int((r==3).sum()),'ear_ok':int((er==0).sum()),'ear_front_hard':int((er==4).sum()),'ear_collision':int((er==3).sum())}
 (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
