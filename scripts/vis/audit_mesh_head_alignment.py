"""核对保存的 GLB 变换、网格头模切片和已有 front 标定候选。"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.template_identity import load_template_identity, sha256_file
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth
from scripts.recon_3d.recon3D import load_calib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.output_dir.resolve().is_relative_to((args.data_dir / 'pde_governance').resolve()):
        raise ValueError('输出必须按 Image ID 保存于 pde_governance')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(args.head)
    base = args.data_dir / 'pixal3d'
    glb = base / f'{args.data_dir.name}.glb'
    hair_path = base / 'hair_mesh_aligned_best.obj'
    transform_path = base / 'glb_to_world.npz'
    raw = o3d.io.read_triangle_mesh(str(glb))
    hair = o3d.io.read_triangle_mesh(str(hair_path))
    with np.load(transform_path) as t:
        similarity = np.eye(4)
        similarity[:3, :3] = float(t['umeyama_scale'].item()) * t['umeyama_R']
        similarity[:3, 3] = t['umeyama_t']
        transform = t['icp_T'] @ similarity
    transformed = np.c_[np.asarray(raw.vertices), np.ones(len(raw.vertices))] @ transform.T
    world = transformed[:, :3] / transformed[:, 3:]
    current = np.asarray(hair.vertices)
    rng = np.random.default_rng(42)
    pick = rng.choice(len(world), min(20000, len(world)), replace=False)
    distance = cKDTree(current).query(world[pick])[0]
    reverse_pick = rng.choice(len(current), min(20000, len(current)), replace=False)
    reverse_distance = cKDTree(world).query(current[reverse_pick])[0]
    with np.load(args.head) as h:
        head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(h['vertices']),
                                       o3d.utility.Vector3iVector(h['faces']))
    reference = o3d.io.read_triangle_mesh('data/head_model.obj')
    report = {'template_identity': identity, 'seed': 42,
              'input_sha256': {str(f.resolve()): sha256_file(f) for f in [glb, hair_path, transform_path]},
              'raw_to_aligned_vertex_distance_mm_q50_q99_max': (np.quantile(distance, [.5, .99, 1]) * 1000).tolist(),
              'aligned_to_raw_vertex_distance_mm_q50_q99_max': (np.quantile(reverse_distance, [.5, .99, 1]) * 1000).tolist(),
              'vertices': {'raw': len(world), 'aligned': len(current)},
              'calibrations': {},
              'limits': ['相机候选仅诊断，不覆盖 front.npy；glb_param 历史来源缺少独立留出验证。',
                         '原始网格与已对齐网格一致，只能排除保存变换失配，不能证明 ICP 配准正确。',
                         '切片由顶点距平面 2mm 内的散点表示，不是封闭截面或体积真值。']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for mesh, label, color in [(hair, 'Aligned Pixal3D mesh', '#2a9d8f'),
                                (reference, 'Reconstruction head', '#f4a261'),
                                (head, 'Blender template head', '#e63946')]:
        v = np.asarray(mesh.vertices)
        for ax, slice_axis, horizontal in [(axes[0], 0, 2), (axes[1], 2, 0)]:
            keep = (np.abs(v[:, slice_axis]) < .002) & (v[:, 1] > 1.68)
            ax.scatter(v[keep, horizontal] * 1000, v[keep, 1] * 1000, s=2, label=label, c=color)
            ax.set_aspect('equal')
            ax.set_xlabel('Z (mm)' if horizontal == 2 else 'X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.grid(alpha=.2)
    axes[0].set_title('Side profile: |X| < 2 mm')
    axes[1].set_title('Front profile: |Z| < 2 mm')
    axes[0].legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'mesh_head_slices.png', dpi=170)
    plt.close(fig)
    seg = cv2.imread(str(args.data_dir / 'maps/seg/front.png'), 0) > 127
    photo = cv2.resize(cv2.imread(str(args.data_dir / 'raw_img.png')), seg.shape[::-1])
    for name in ['front.npy', 'glb_param.npy', 'front_dense_silhouette.npy']:
        path = args.data_dir / 'maps/param' / name
        if not path.exists():
            report['calibrations'][name] = {'status': 'missing'}
            continue
        camera = load_calib(str(path)).numpy()
        chart = VisibleSurfaceGrowth(camera, seg.astype(np.int32), np.zeros((*seg.shape, 2)), hair)
        head_chart = VisibleSurfaceGrowth(camera, seg.astype(np.int32), np.zeros((*seg.shape, 2)), head)
        row_axes = camera[:2, :3].copy()
        row_axes /= np.linalg.norm(row_axes, axis=1, keepdims=True)
        report['calibrations'][name] = {
            'sha256': sha256_file(path), 'seg_pixels': int(seg.sum()),
            'mesh_hit_in_seg_pixels': int((chart.head_hit & seg).sum()),
            'mesh_behind_template_in_seg_pixels': int((seg & chart.head_hit & head_chart.head_hit &
                                                       (chart.head_z < head_chart.head_z - 1e-5)).sum()),
            'image_axis_angle_deg': float(np.degrees(np.arccos(np.clip(row_axes[0] @ row_axes[1], -1, 1))))}
        image = photo.copy()
        for mask, color in [(seg, (0, 255, 0)), (chart.head_hit, (255, 255, 0))]:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1)
        cv2.imwrite(str(args.output_dir / f'{path.stem}_mesh_outline.png'), image)
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
