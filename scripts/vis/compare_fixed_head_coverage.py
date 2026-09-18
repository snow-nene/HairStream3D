"""固定头模和观测相机，比较中心线覆盖代理；不代替实际渲染。"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按 Image ID 聚合')
    z = np.load(a.head)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(z['vertices']),
                                   o3d.utility.Vector3iVector(z['faces']))
    report = {'metric': '固定头模可见目标的像素中心线代理，非渲染面积', 'views': {}}
    for view in ['front', 'left', 'right', 'back']:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        chart = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view), seg.astype(int),
                                    np.zeros((*seg.shape, 2)), mesh)
        report['views'][view] = {}
        for run in a.runs:
            with np.load(run/'all_root_prefixes.npz') as packed:
                covered = chart.covered(packed['strands'])
            report['views'][view][run.name] = {'gap_pixels': int((chart.target & ~covered).sum()),
                                              'target_pixels': int(chart.target.sum())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
