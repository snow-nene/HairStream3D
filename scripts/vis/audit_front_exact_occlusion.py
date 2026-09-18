"""比较原图 front 像素深度守卫与候选点精确射线遮挡。"""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--repair-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(a.repair_dir/'geometry/template_head/template_head.npz')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),
                                   o3d.utility.Vector3iVector(data['faces']))
    mesh.compute_vertex_normals()
    points = data['vertices'] + .0008*np.asarray(mesh.vertex_normals)
    seg = cv2.imread(str(a.data_dir/'maps/seg/front.png'), 0) > 127
    camera = load_observation_camera(a.data_dir, 'front')
    chart = VisibleSurfaceGrowth(camera, seg.astype(int), np.zeros((*seg.shape, 2)), mesh)
    ids, depth, valid = chart.pixels(points)
    ix = np.flatnonzero(valid)
    x, y = ids[valid].T
    old_visible = np.zeros(len(points), bool)
    old_visible[ix] = ~chart.head_hit[y, x] | (depth[valid] >= chart.head_z[y, x]-1e-5)
    positive = np.zeros(len(points), bool)
    positive[ix] = seg[y, x]
    # 正交相机每个点使用自身精确连续像素射线；不依赖像素中心缓存。
    if not np.allclose(camera[3], [0, 0, 0, 1]):
        raise ValueError('此审计要求正交 front 相机')
    toward_camera = np.linalg.inv(camera)[:3, 2]
    toward_camera /= np.linalg.norm(toward_camera)
    origins = points + 2*toward_camera
    directions = np.broadcast_to(-toward_camera, points.shape)
    hit = chart.scene.cast_rays(o3d.core.Tensor(np.c_[origins, directions].astype(np.float32)))['t_hit'].numpy()
    occlusion_margin = 2-hit
    exact_hidden = occlusion_margin > .00001
    old_reject = valid & old_visible & ~positive
    false_reject = old_reject & exact_hidden
    report = {'candidate_count': len(points), 'old_front_rejected': int(old_reject.sum()),
              'exact_hidden_among_rejected': int(false_reject.sum()),
              'hidden_margin_over_0_5mm': int((false_reject & (occlusion_margin > .0005)).sum()),
              'hidden_margin_over_2mm': int((false_reject & (occlusion_margin > .002)).sum()),
              'limit': '全头模偏移0.8mm候选根；不是已停止发丝的首次失败点。遮挡仅相对当前头模成立。'}
    left_seg = cv2.imread(str(a.data_dir/'maps/seg/left.png'), 0) > 127
    left = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, 'left'),
        left_seg.astype(int), np.zeros((*left_seg.shape, 2)), mesh)
    with np.load(a.repair_dir/'ablation_side/all_root_prefixes.npz') as packed:
        strands = packed['strands']
        covered = left.covered(strands)
    gap = left.target & ~covered
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), 8)
    largest = labels == 1+np.argmax(stats[1:, cv2.CC_STAT_AREA]) if count > 1 else gap
    lp, _, lv = left.pixels(points)
    in_gap = np.zeros(len(points), bool)
    in_gap[lv] = largest[lp[lv, 1], lp[lv, 0]]
    diagnostics = np.load(a.repair_dir/'ablation_side/root_diagnostics.npz')
    selected = diagnostics['vertex_ids']
    selected_gap = in_gap[selected]
    stops = diagnostics['stop_code'][selected_gap]
    report['left_largest_gap'] = {
        'pixels': int(largest.sum()), 'projected_candidates': int(in_gap.sum()),
        'root_front_rejected': int((in_gap & old_reject).sum()),
        'root_false_rejected': int((in_gap & false_reject).sum()),
        'selected_roots': int(selected_gap.sum()),
        'selected_roots_front_rejected_at_start': int(old_reject[selected[selected_gap]].sum()),
        'selected_stop_codes': {str(int(k)): int(v) for k, v in zip(*np.unique(stops, return_counts=True))},
        'note': '已选根先通过守卫；stop_code=4表示后续生长停止，不等于根点语义不合法。'}
    cv2.imwrite(str(a.output_dir/'left_largest_gap.png'), largest.astype(np.uint8)*255)
    if a.trace_dir:
        trace = np.load(a.trace_dir/'root_diagnostics.npz')
        choose = in_gap[trace['vertex_ids']] & (trace['integration_stop_code'] == 4)
        failures = trace['failure_points'][choose]
        if not np.isfinite(failures).all():
            raise ValueError('语义停止缺少首次失败位置')
        fp, _, fv = chart.pixels(failures)
        rays = np.c_[failures+2*toward_camera, np.broadcast_to(-toward_camera, failures.shape)]
        fh = chart.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        margins = 2-fh
        report['left_growth_failure'] = {
            'count': len(failures), 'exact_hidden': int((margins > .00001).sum()),
            'exact_hidden_over_2mm': int((margins > .002).sum()),
            'length_mm_quantiles': np.quantile(trace['length_m'][choose]*1000, [0,.25,.5,.75,1]).tolist()}
        raw = cv2.imread(str(a.data_dir/'raw_img.png'))
        for pixel, inside, hidden in zip(fp, fv, margins > .00001):
            if inside:
                cv2.circle(raw, tuple(pixel), 2, (0,255,0) if hidden else (0,0,255), -1)
        cv2.imwrite(str(a.output_dir/'left_failure_front_overlay.png'), raw)
    np.savez_compressed(a.output_dir/'point_audit.npz', points=points, pixels=ids,
                        old_reject=old_reject, exact_hidden=exact_hidden, occlusion_margin=occlusion_margin)
    (a.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
