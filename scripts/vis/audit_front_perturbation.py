"""固定结果，对front相机平移及蒙版边界扰动做语义敏感度审计。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth,load_observation_camera,sample_polyline
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按Image ID聚合')
    d=np.load(a.head)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),o3d.utility.Vector3iVector(d['faces']))
    camera=load_observation_camera(a.data_dir,'front')
    seg=cv2.imread(str(a.data_dir/'maps/seg/front.png'),0)>127
    samples={}
    for run in a.runs:
        with np.load(run/'all_root_prefixes.npz') as packed:
            samples[run.name]=np.concatenate([sample_polyline(s,.0005) for s in packed['strands']])
    report={'limit':'固定发丝，0.5mm采样；测量输入敏感度，不代表完整重跑稳定性或跨样本泛化。','variants':{}}
    for name,dx,dy,morph in [('original',0,0,0),('x_minus_1px',-1,0,0),('x_plus_1px',1,0,0),
                             ('y_minus_1px',0,-1,0),('y_plus_1px',0,1,0),('erode_1px',0,0,-1),('dilate_1px',0,0,1)]:
        c=camera.copy(); c[0,3]+=2*dx/(seg.shape[1]-1); c[1,3]+=2*dy/(seg.shape[0]-1)
        mask=seg.astype(np.uint8)
        if morph: mask=(cv2.erode if morph<0 else cv2.dilate)(mask,np.ones((3,3),np.uint8))
        chart=VisibleSurfaceGrowth(c,mask,np.zeros((*seg.shape,2)),mesh)
        guard=SubjectGuards({'front':chart},side_veto=False)
        report['variants'][name]={}
        for run,points in samples.items():
            bad=sum(int(np.isin(guard.points(points[i:i+65536],collision=False),[4,5]).sum()) for i in range(0,len(points),65536))
            report['variants'][name][run]={'samples':len(points),'violations':bad,'fraction':bad/len(points)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
