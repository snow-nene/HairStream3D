"""固定缓存上比较三种免训练 2D->3D 方向提升模式。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.multiview_direction_lifting import lift_visible_directions, compare_axial_directions


def run(args):
    base, cache_dir, out = args.data_dir, args.cache_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / f'{args.view}_calibration_samples.npz'
    map_path = base / 'maps/strand_map' / f'{args.view}.png'
    seg_path = base / 'maps/seg' / f'{args.view}.png'
    cache = np.load(sample_path)
    strand = cv2.cvtColor(cv2.imread(str(map_path)), cv2.COLOR_BGR2RGB)
    seg = cv2.imread(str(seg_path), 0) > 127
    y, x, world, camera = cache['pixel_y'], cache['pixel_x'], cache['mesh_world'], cache['camera']
    labels = np.ones(len(world), dtype=np.int32)
    # The ablation measures lifting itself; labels are deliberately local to
    # this visible sample set and do not claim a completed volume partition.
    unit = strand.astype(float) / 255.0
    dx = 1 - 2 * unit[y, x, 2]
    dy = 2 * unit[y, x, 1] - 1
    inverse = np.linalg.inv(camera)[:3, :3]
    base_tangent = np.column_stack([dx * 2 / (seg.shape[1]-1),
                                    dy * 2 / (seg.shape[0]-1),
                                    np.zeros(len(y))]) @ inverse.T
    normals = cache.get('mesh_normals', None)
    if normals is None:
        # Older calibration caches have no normals; use the cached visible
        # normal-free 2D ray tangent as the reference and mark provenance.
        tangent = base_tangent
        normal_source = 'not_cached; image-ray tangent reference'
    else:
        ray = inverse[:, 2]
        denominator = normals @ ray
        tangent = base_tangent.copy()
        good = np.abs(denominator) > 1e-8
        tangent[good, 2] = -np.sum(normals[good] * base_tangent[good], axis=1) / denominator[good]
        normal_source = 'cached_mesh_normals'
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    differential = lift_visible_directions(y, x, world, labels, strand, seg.shape,
                                            step_px=args.step_px)
    consistency = compare_axial_directions(differential['directions'], tangent,
                                            differential['accepted'], args.max_angle_deg)
    modes = {
        'mesh_tangent': np.ones(len(y), dtype=bool),
        'visible_intersection_difference': differential['accepted'],
        'difference_with_tangent_gate': consistency['compatible'],
    }
    report = {
        'status': 'diagnostic_only', 'view': args.view,
        'normal_reference': normal_source, 'step_px': args.step_px,
        'max_angle_deg': args.max_angle_deg, 'samples': int(len(y)),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (sample_path, map_path, seg_path)},
        'modes': {},
    }
    for name, accepted in modes.items():
        if name == 'mesh_tangent':
            angles = np.zeros(len(y), dtype=float)
        else:
            angles = consistency['angle_deg']
        report['modes'][name] = {
            'accepted': int(accepted.sum()),
            'acceptance_fraction': float(accepted.mean()),
            'angle_median_deg': float(np.median(angles[accepted])) if accepted.any() else None,
            'angle_q95_deg': float(np.quantile(angles[accepted], .95)) if accepted.any() else None,
            'direction_norm_median': float(np.median(np.linalg.norm(
                (tangent if name == 'mesh_tangent' else differential['directions'])[accepted], axis=1)))
            if accepted.any() else None,
        }
    np.savez_compressed(out / f'{args.view}_lifting_ablation.npz',
                        tangent=tangent.astype(np.float32),
                        differential=differential['directions'],
                        differential_accepted=differential['accepted'],
                        consistency_accepted=consistency['compatible'],
                        angle_deg=consistency['angle_deg'])
    report['rejection_reasons'] = {str(k): int(v) for k, v in
        zip(*np.unique(differential['reason'], return_counts=True))}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--view', default='front')
    parser.add_argument('--step-px', type=int, default=2)
    parser.add_argument('--max-angle-deg', type=float, default=30.)
    run(parser.parse_args())
