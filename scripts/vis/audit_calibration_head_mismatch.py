"""固定左侧缺口射线，隔离标定头模和渲染头模的遮挡差异。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--repair-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按Image ID聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data=np.load(a.repair_dir/'feasible_space_20260911/feasible_space.npz')
    source=o3d.io.read_triangle_mesh('data/head_model.obj')
    vertices=np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    render=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices['vertices']),
                                   o3d.utility.Vector3iVector(vertices['faces']))
    seg=cv2.imread(str(a.data_dir/'maps/seg/front.png'),0)>127
    cam=load_observation_camera(a.data_dir,'front')
    report={}
    for name,mesh in [('render_head',render),('calibration_head',source)]:
        chart=VisibleSurfaceGrowth(cam,seg.astype(int),np.zeros((*seg.shape,2)),mesh)
        guards=SubjectGuards({'front':chart},side_veto=False)
        feasible=np.zeros(len(data['surface']),bool)
        for offset in data['offsets']:
            points=data['surface']+offset*data['toward_camera']
            codes=guards.points(points,collision=False)
            signed=chart.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)),nsamples=3).numpy()
            feasible |= (codes==0)&(signed>=.0004)
        np.save(a.output_dir/f'{name}_feasible.npy',feasible)
        report[name]={'fixed_gap_rays':len(feasible),'feasible_rays':int(feasible.sum()),
                      'front_head_silhouette_pixels':int(chart.head_hit.sum())}
    report['limit']='几何隔离实验，不代表标定头模是真实头皮；保持缺口射线固定，不改变front.npy。'
    (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
