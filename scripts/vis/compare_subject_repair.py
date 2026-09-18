"""固定原图与侧视先验，比较修复前后形状、可见语义与头模兼容性。"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.template_identity import load_template_identity, require_bound_template, sha256_file
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera, sample_polyline
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    for name in ['before', 'after']:
        p.add_argument(f'--{name}-input', type=Path, required=True)
        p.add_argument(f'--{name}-head', type=Path, required=True)
        p.add_argument(f'--{name}-render', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按 Image ID 聚合')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    full = o3d.io.read_triangle_mesh(str(a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'))
    report = {'limits': ['中心线覆盖不等于实际渲染面积', '非front仅为Flux先验',
                         '隐藏头皮是拟合先验，非真实人体扫描'], 'runs': {}}
    fixed_top = None
    for name in ['before', 'after']:
        source, head_path = getattr(a, f'{name}_input'), getattr(a, f'{name}_head')
        identity = load_template_identity(head_path)
        with np.load(source) as d:
            require_bound_template(d, Path(identity['template_path']))
            strands = [s[np.r_[True, np.linalg.norm(np.diff(s, axis=0), axis=1)>0]] for s in d['strands']]
            source_ids = d['source_root_ids'] if 'source_root_ids' in d else np.arange(len(strands))
        with np.load(head_path) as d:
            vertices, faces = d['vertices'], d['faces']
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        result = {'source_sha256': sha256_file(source), 'template_identity': identity,
                  'strand_count': len(strands), 'views': {}, 'groups': {}}
        charts = {}
        for view in ['front', 'left', 'right', 'back', 'top']:
            if view == 'top':
                if fixed_top is None:
                    center = (vertices.min(0)+vertices.max(0))/2
                    scale = 1.8/max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 2]))
                    camera = np.array([[scale, 0, 0, -scale*center[0]], [0, 0, scale, -scale*center[2]],
                                       [0, 1, 0, -center[1]], [0, 0, 0, 1]])
                    preliminary = VisibleSurfaceGrowth(camera, np.ones((512, 512), int), np.zeros((512, 512, 2)), mesh)
                    seg = preliminary.target & (preliminary.head_points[:, 1].reshape(512, 512)>vertices[:, 1].max()-.2*np.ptp(vertices[:, 1]))
                    seg &= preliminary.normals[:, 1].reshape(512, 512)>.5
                    fixed_top = camera, seg
                camera, seg = fixed_top
            else:
                seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
                camera = load_observation_camera(a.data_dir, view)
            chart = VisibleSurfaceGrowth(camera, seg.astype(int), np.zeros((*seg.shape, 2)), mesh)
            covered = chart.covered(strands)
            item = {'fixed_target_pixels': int(seg.sum()), 'covered_target_pixels': int((seg & covered).sum()),
                    'gap_fraction': float((seg & ~covered).sum()/max(seg.sum(), 1)),
                    'visible_pixels_outside_seg': int((~seg & covered).sum()) if view != 'top' else None}
            if view != 'top':
                charts[view] = chart
                hair = VisibleSurfaceGrowth(camera, seg.astype(int), np.zeros((*seg.shape, 2)), full)
                points = hair.head_points[(seg & hair.head_hit).ravel()]
                signed = chart.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)), nsamples=11).numpy()
                item['mesh_candidate_inside_head_fraction'] = float((signed < -.0004).mean())
            overlay = np.zeros((*seg.shape, 3), np.uint8)
            overlay[seg & ~covered] = [0, 0, 255]
            overlay[seg & covered] = [0, 200, 0]
            overlay[~seg & covered] = [255, 0, 255]
            if view == 'front':
                photo = cv2.resize(cv2.imread(str(a.data_dir/'raw_img.png')), seg.shape[::-1])
                overlay = cv2.addWeighted(photo, .55, overlay, .45, 0)
            cv2.imwrite(str(a.output_dir/f'{name}_{view}_coverage.png'), overlay)
            result['views'][view] = item
        guards = SubjectGuards(charts)
        lengths, turns, body, owners = [], [], [], []
        semantic_bad = []
        for i, strand in enumerate(strands):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
            grid = np.arange(0., arc[-1], .002)
            sampled = np.column_stack([np.interp(grid, arc, strand[:, k]) for k in range(3)])
            vector = np.diff(sampled, axis=0)
            vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
            angle = np.degrees(np.arccos(np.clip((vector[:-1]*vector[1:]).sum(1), -1, 1)))
            turns.append(float(angle.max(initial=0)))
            lengths.append(np.linalg.norm(np.diff(strand, axis=0), axis=1).sum())
            body.extend(sampled[3::2])
            owners.extend([i]*len(sampled[3::2]))
            if name == 'after':
                codes = guards.points(sample_polyline(strand, .000125), collision=False)
                if np.isin(codes, [4, 5, 6]).any():
                    semantic_bad.append(i)
        body, owners = np.asarray(body), np.asarray(owners)
        separation = charts['front'].scene.compute_distance(o3d.core.Tensor(body.astype(np.float32))).numpy()
        lengths, turns = np.asarray(lengths), np.asarray(turns)
        for group, selected in [('baseline', (source_ids >= 0) & (source_ids < 9289)),
                                ('supplemental', (source_ids < 0) | (source_ids >= 9289))]:
            samples = separation[selected[owners]]
            result['groups'][group] = {'strands': int(selected.sum()),
                'length_mm_q10_q50_q90': (np.quantile(lengths[selected], [.1, .5, .9])*1000).tolist(),
                'at_120mm': int((np.abs(lengths[selected]-.12)<.00001).sum()),
                'turn_over_60deg_at_2mm': int((turns[selected]>60).sum()),
                'body_distance_median_mm': float(np.median(samples)*1000),
                'body_within_4mm_fraction': float((samples<.004).mean())}
        if name == 'after':
            result['strict_semantic_violation_strand_ids'] = semantic_bad
        report['runs'][name] = result
        print(json.dumps({name: result}), flush=True)
    (a.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.3))
    for row, name in enumerate(['before', 'after']):
        for col, view in enumerate(['front', 'left', 'right', 'back']):
            path = getattr(a, f'{name}_render')/f'{view}.png'
            axes[row, col].imshow(plt.imread(path))
            axes[row, col].set_title(f'{name}: {view}')
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(a.output_dir/'render_comparison.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
