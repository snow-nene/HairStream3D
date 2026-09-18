"""将壳层多视角证据扩散到头模表面，评估局部连续区域。"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import label
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import load_observation_camera
from lib.multiview_hair_evidence import diffuse_surface_evidence

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--shell-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 h=np.load(a.data_dir/'pde_governance/volume_partition_integration/subject_repair_v32/geometry/template_head/template_head.npz');v,f=h['vertices'],h['faces'];edges=np.unique(np.sort(np.vstack([f[:,:2],f[:,1:],f[:,[2,0]]]),axis=1),axis=0)
 shell=o3d.io.read_triangle_mesh(str(a.shell_dir/'hair_outer_surface.obj'));sv=np.asarray(shell.vertices);sf=np.asarray(shell.triangles);cent=sv[sf].mean(1);tree=cKDTree(v);anchors=tree.query(cent)[1]
 scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(shell));scores=np.zeros(len(cent))
 for view,weight in [('front',3.),('left',1.)]:
  cam=load_observation_camera(a.data_dir,view);seg=cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'),0)>127;q=np.c_[cent,np.ones(len(cent))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ii=np.rint(uv).astype(int);valid=((ii>=0)&(ii<512)).all(1);hair=np.zeros(len(cent),bool);hair[valid]=seg[ii[valid,1],ii[valid,0]];toward=np.linalg.inv(cam)[:3,2];toward/=np.linalg.norm(toward);hit=scene.cast_rays(o3d.core.Tensor(np.c_[cent+2*toward,np.broadcast_to(-toward,cent.shape)].astype(np.float32)))['t_hit'].numpy()>=1.999;visible=valid&hit
  scores += weight*np.where(visible&hair,1.,np.where(visible&~hair,-1.,0.))
 # 以面中心分数作为头模顶点锚点，取最高证据覆盖。
 order=np.argsort(np.abs(scores))[::-1];chosen=[];seen=set()
 for i in order:
  aidx=int(anchors[i])
  if aidx not in seen:seen.add(aidx);chosen.append((aidx,scores[i]))
 ai=np.array([x[0] for x in chosen]);av=np.array([x[1] for x in chosen]);field=diffuse_surface_evidence(v,edges,ai,av,smoothness=20.)
 report={'shell_faces':len(sf),'head_vertices':len(v),'anchors':len(ai),'raw_score_q10_q50_q90':np.quantile(scores,[.1,.5,.9]).tolist(),'field_q10_q50_q90':np.quantile(field,[.1,.5,.9]).tolist(),'thresholds':{}}
 # 耳前像素框对应头模投影顶点，检查连续候选数量。
 cam=load_observation_camera(a.data_dir,'left');q=np.c_[v,np.ones(len(v))]@cam.T;uv=(q[:,:2]/q[:,3:]+1)*255.5;ear=(uv[:,0]>=195)&(uv[:,0]<238)&(uv[:,1]>=167)&(uv[:,1]<207)
 for t in [.2,.5,1.,1.5,2.]:
  active=field>=t;sub=np.zeros(len(v),bool);sub[active]=1;report['thresholds'][str(t)]={'active_vertices':int(active.sum()),'ear_active_vertices':int((active&ear).sum())}
 np.savez_compressed(a.output_dir/'evidence_field.npz',vertices=v,faces=f,field=field,ear_mask=ear)
 (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
