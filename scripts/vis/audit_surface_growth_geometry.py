#!/usr/bin/env python3
"""无需渲染的多视角覆盖审计；左右按各图像坐标，非正面仅为补全先验。"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import (
    VisibleSurfaceGrowth, load_observation_camera, sample_polyline, audit_strands,
)
from lib.template_identity import load_template_identity, require_bound_template


def regional_metrics(target, covered):
    """分区相对目标包围盒，避免按整张图三等分遗漏头顶或画面左侧。"""
    yy, xx = np.indices(target.shape)
    y, x = np.nonzero(target)
    if not len(x):
        return {}
    masks = {'all': target,
             'image_left': target & (xx < (x.min() + x.max() + 1) / 2),
             'image_right': target & (xx >= (x.min() + x.max() + 1) / 2),
             'upper_third': target & (yy < y.min() + (y.max()-y.min()+1)/3)}
    result = {}
    for name, mask in masks.items():
        gap = mask & ~covered
        _, _, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), 8)
        result[name] = {'target_pixels': int(mask.sum()), 'gap_pixels': int(gap.sum()),
                        'gap_fraction': float(gap.sum()/max(1, mask.sum())),
                        'largest_gap_pixels': int(stats[1:, cv2.CC_STAT_AREA].max(initial=0))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--views', nargs='+', choices=['front', 'left', 'right', 'back', 'top'],
                        default=['front', 'left', 'right', 'back', 'top'])
    parser.add_argument('--collision-audit', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(args.head)
    with np.load(args.input) as data:
        require_bound_template(data, Path(identity['template_path']))
        strands = [p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1)>0]] for p in data['strands']]
    with np.load(args.head) as data:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),
                                        o3d.utility.Vector3iVector(data['faces']))
    roots = cKDTree(np.array([s[0] for s in strands]))
    samples = np.concatenate([sample_polyline(s, .0005) for s in strands])
    tree = cKDTree(samples)
    lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strands])
    report = {'input': str(args.input.resolve()), 'template_identity': identity,
              'orientation': 'image_left/right are screen coordinates, not anatomical sides',
              'metric': 'head-depth-tested centreline pixels; no strand thickness or photorealistic rendering',
              'strand_count': len(strands), 'length_m_quantiles': np.quantile(lengths,[0,.1,.5,.9,1]).tolist(),
              'zero_length_count': int((lengths==0).sum()), 'views': {}}
    front_chart = None
    for view in args.views:
        if view == 'top':
            vertices = np.asarray(mesh.vertices)
            center = (vertices.min(0) + vertices.max(0))/2
            scale = 1.8/max(np.ptp(vertices[:,0]),np.ptp(vertices[:,2]))
            camera = np.array([[scale,0,0,-scale*center[0]],
                               [0,0,scale,-scale*center[2]], [0,1,0,-center[1]], [0,0,0,1]])
            seg = np.ones((512,512),bool)
        else:
            seg = cv2.imread(str(args.data_dir/'maps/seg'/f'{view}.png'),0)
            if seg is None:
                raise ValueError(f'missing segmentation: {view}')
            seg = seg > 127
            camera = load_observation_camera(args.data_dir, view)
        chart = VisibleSurfaceGrowth(camera, seg.astype(np.int32),np.zeros((*seg.shape,2)),mesh)
        if view == 'front':
            front_chart = chart
        if view == 'top':
            # Fixed geometric crown proxy, independent of existing hair.
            chart.target &= (chart.head_points[:,1].reshape(seg.shape) >
                             vertices[:,1].max()-.2*np.ptp(vertices[:,1]))
            chart.target &= chart.normals[:,1].reshape(seg.shape) > .5
            seg = chart.target.copy()
            np.savez_compressed(args.output_dir/'top_camera.npz', camera=camera,
                                image_shape=np.array(seg.shape), target_mask=seg,
                                template_blend_sha256=np.array(identity['template_sha256']))
        covered = chart.covered(strands)
        ids = np.flatnonzero(chart.target)
        points = chart.head_points[ids]
        root_dist = roots.query(points)[0]
        hair_dist = tree.query(points)[0]
        gap = ~covered.ravel()[ids]
        metrics = {'evidence': 'geometric_crown_proxy' if view=='top' else ('original_photo' if view=='front' else 'flux_prior'),
                   'regions': regional_metrics(chart.target,covered),
                   'segmentation_regions': regional_metrics(seg,covered),
                   'visible_head_pixels_outside_seg_with_hair': int((chart.head_hit & ~seg & covered).sum()) if view != 'top' else None,
                   'gap_farther_than_15mm_from_roots': int((gap & (root_dist>=.015)).sum()),
                   'surface_distance_m_quantiles': np.quantile(hair_dist,[0,.5,.9,.99,1]).tolist() if len(ids) else [],
                   'surface_fraction_within_mm': {str(mm): float((hair_dist<=mm/1000).mean()) if len(ids) else None for mm in [1,2,4,8]}}
        gap_mask = (chart.target & ~covered).astype(np.uint8)
        gap_radius = cv2.distanceTransform(gap_mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        metrics['maximum_gap_inscribed_radius_pixels'] = float(gap_radius.max())
        if view == 'top':
            metrics['maximum_gap_inscribed_radius_m'] = float(gap_radius.max()*2/(scale*(seg.shape[1]-1)))
        if view != 'front':
            if front_chart is None:
                front_seg = cv2.imread(str(args.data_dir/'maps/seg/front.png'),0)>127
                front_chart = VisibleSurfaceGrowth(load_observation_camera(args.data_dir,'front'),
                    front_seg.astype(np.int32),np.zeros((*front_seg.shape,2)),mesh)
            pixels, depth, valid = front_chart.pixels(points + .0008*chart.normals[ids])
            conflict = np.zeros(len(ids),bool)
            x,y = pixels[valid].T
            conflict[valid] = (front_chart.head_hit[y,x] &
                (depth[valid] >= front_chart.head_z[y,x]-1e-5) & ~front_chart.hair_domain[y,x])
            metrics['root_launch_conflicts_with_front_semantics'] = int(conflict.sum())
            metrics['gap_root_launch_conflicts_with_front_semantics'] = int((conflict & gap).sum())
        overlay = np.zeros((*seg.shape,3),np.uint8)
        overlay[seg] = (50,50,50)
        overlay[chart.target & ~covered] = (0,0,255)
        overlay[chart.target & covered] = (0,200,0)
        if view != 'top':
            overlay[chart.head_hit & ~seg & covered] = (255,0,255)
        cv2.imwrite(str(args.output_dir/f'{view}_diagnostic.png'),overlay)
        report['views'][view] = metrics
        print(json.dumps({view:metrics}),flush=True)
    if args.collision_audit:
        report['collision_audit'] = audit_strands(strands,mesh)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
