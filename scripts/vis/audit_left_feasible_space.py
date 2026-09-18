"""在左侧缺口射线上审计原图约束下的外部可行空间。"""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--repair-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按Image ID聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),
                                   o3d.utility.Vector3iVector(d['faces']))
    charts = {}
    for view in ['front', 'left']:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        charts[view] = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view),
            seg.astype(int), np.zeros((*seg.shape, 2)), mesh)
    left = charts['left']
    with np.load(a.run_dir/'all_root_prefixes.npz') as packed:
        gap = left.target & ~left.covered(packed['strands'])
    indices = np.flatnonzero(gap)
    surface = left.head_points[indices]
    toward = left.inverse[:3, 2].copy()
    toward /= np.linalg.norm(toward)
    offsets = np.linspace(.0008, .05, 100)
    guard = SubjectGuards({'front': charts['front']}, side_veto=False)
    feasible = np.zeros((len(surface), len(offsets)), bool)
    for i, offset in enumerate(offsets):
        points = surface+offset*toward
        codes = guard.points(points, collision=False)
        signed = left.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)), nsamples=3).numpy()
        feasible[:, i] = (codes == 0) & (signed >= .0004)
    report = {'gap_pixels': len(indices), 'offset_range_mm': [0.8, 50],
              'offset_sample_count': len(offsets), 'feasible_by_max_offset_mm': {
                  str(mm): int(feasible[:, offsets <= mm/1000].any(1).sum()) for mm in [2,5,10,20,50]},
              'infeasible_sampled_rays': int((~feasible.any(1)).sum()),
              'limit': '固定左侧像素中心射线、当前头模与原图标定；有限离散采样不是连续不可行证明。'}
    colors = np.zeros((*gap.shape, 3), np.uint8)
    colors.reshape(-1,3)[indices] = [0,0,255]
    colors.reshape(-1,3)[indices[feasible.any(1)]] = [0,255,0]
    cv2.imwrite(str(a.output_dir/'feasible_left.png'), colors)
    np.savez_compressed(a.output_dir/'feasible_space.npz', pixel_indices=indices,
        surface=surface, offsets=offsets, toward_camera=toward, feasible=feasible)
    (a.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
