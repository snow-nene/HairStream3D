"""全包围盒四类体积证据原型；冲突退回未知，不执行标签优化。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration


def build_volume_evidence_prototype():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    parent = base / 'pde_governance/volume_domain_audit'
    out = parent / 'step_06_four_class_evidence'
    out.mkdir(parents=True, exist_ok=True)
    bounds = json.loads((parent / 'step_01_sections/report.json').read_text())
    low, high = np.array(bounds['bounds_min']), np.array(bounds['bounds_max'])
    shape = (64, 64, 64)
    axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
    spacing = (high-low)/63
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    nearest = scene.compute_distance(o3d.core.Tensor(points)).numpy()
    wrap = o3d.io.read_triangle_mesh(str(parent / 'step_04_root_ray_consistency/full_model_outer_wrap_large.off'))
    assert wrap.is_watertight()
    wrap_scene = o3d.t.geometry.RaycastingScene()
    wrap_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wrap))
    wrap_distance = wrap_scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=5).numpy()
    head_paths = ['data/head_model.obj', str(base / 'pde_governance/flame_head_fit/step_01_face_shape_fit/fitted_head.obj')]
    head_core = np.ones(len(points), bool)
    for path in head_paths:
        head = o3d.io.read_triangle_mesh(path)
        head_scene = o3d.t.geometry.RaycastingScene()
        head_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head))
        signed = head_scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=11).numpy()
        head_core &= signed < -.015
    head_core &= wrap_distance < -.006
    hair_votes, free_votes = np.zeros(len(points), np.uint8), np.zeros(len(points), np.uint8)
    front_hair = np.zeros(len(points), bool)
    per_view = {}
    for view in ['front', 'left', 'right', 'back']:
        calib = (load_calib(str(base / 'maps/param/front_dense_silhouette.npy')).numpy() if view == 'front'
                 else load_blender_view_calibration(str(base), view)[0]).astype(float)
        mask = cv2.imread(str(base / 'maps/seg' / f'{view}.png'), 0) > 127
        interior = distance_transform_edt(mask)
        clip = np.column_stack([points, np.ones(len(points))])@calib.T
        uv = clip[:, :2]/clip[:, 3:4]
        inverse = np.linalg.inv(calib)
        mesh_clip = np.column_stack([np.asarray(mesh.vertices), np.ones(len(mesh.vertices))])@calib.T
        near_z = mesh_clip[:, 2].max()+1
        origin = np.column_stack([uv, np.full(len(points), near_z), np.ones(len(points))])@inverse.T
        origin = origin[:, :3]/origin[:, 3:4]
        direction = points-origin
        point_distance = np.linalg.norm(direction, axis=1)
        direction /= point_distance[:, None]
        hits = scene.cast_rays(o3d.core.Tensor(np.column_stack([origin, direction]).astype(np.float32)))
        depth = hits['t_hit'].numpy()
        incidence = np.abs((hits['primitive_normals'].numpy()*direction).sum(1))
        reliable = np.isfinite(depth) & (incidence > .4) & (np.abs(uv) <= 1).all(1)
        px, py = (uv[:, 0]+1)/2*(mask.shape[1]-1), (uv[:, 1]+1)/2*(mask.shape[0]-1)
        hair_pixel = map_coordinates(interior, [py, px], order=1, mode='constant', cval=0) >= 3
        delta = point_distance-depth
        hair = reliable & hair_pixel & (delta >= -.004) & (delta <= .008) & (nearest <= .008)
        free = reliable & (delta < -.012) & (nearest > .008)
        hair_votes += hair
        free_votes += free
        if view == 'front':
            front_hair = hair
        per_view[view] = {'surface_hair_candidates': int(hair.sum()), 'free_space_candidates': int(free.sum())}
    hair_evidence = front_hair | (hair_votes >= 2)
    air_evidence = (wrap_distance > .006) | (free_votes >= 2)
    conflict = ((hair_evidence & air_evidence) | (hair_evidence & head_core) | (air_evidence & head_core))
    labels = np.zeros(len(points), np.uint8)
    labels[hair_evidence & ~conflict] = 1
    labels[head_core & ~conflict] = 2
    labels[air_evidence & ~conflict] = 3
    assert (labels[conflict] == 0).all()
    old_sdf = np.load(base / 'pde_governance/current_full_reconstruction_10k_128_20260904/pde_domain_sdf.npy')
    old_idx = ((points-low)/(high-low)*(np.array(old_sdf.shape)-1)).T
    old_domain = map_coordinates(old_sdf, old_idx, order=0, mode='nearest') > 0
    names = ['unknown', 'hair_surface_evidence', 'head_core_prior', 'air_evidence']
    counts = {name: int((labels == i).sum()) for i, name in enumerate(names)}
    report = {'resolution': list(shape), 'bounds_min': low.tolist(), 'bounds_max': high.tolist(),
              'spacing_mm': (spacing*1000).tolist(), 'class_counts': counts,
              'conflict_voxels_returned_to_unknown': int(conflict.sum()), 'per_view': per_view,
              'old_domain_class_counts': {name: int(((labels == i) & old_domain).sum()) for i, name in enumerate(names)},
              'hair_evidence_outside_old_domain': int(((labels == 1) & ~old_domain).sum()),
              'production_ready': False,
              'note': '整包围盒上建立证据，不是旧域裁剪。蓝色为两非水密头模深层一致先验，非真值；侧背来自生成视图，非独立观测。绿色仅可信可见表面种子，不是全部头发体积。黄色未知包括隐藏头发、头部边界、漏射和冲突。紫色仅用于显示冲突，其数据标签仍为未知。未执行图割、PDE或移动发根。'}
    np.savez_compressed(out / 'volume_evidence.npz', labels=labels.reshape(shape), conflict=conflict.reshape(shape),
                        hair_votes=hair_votes.reshape(shape), free_votes=free_votes.reshape(shape),
                        head_core_prior=head_core.reshape(shape), b_min=low, b_max=high)
    display = labels.copy()
    display[conflict] = 4
    display = display.reshape(shape)
    tri = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (axis, value) in zip(axs.ravel(), [(1, 1.70), (1, 1.78), (1, 1.84), (0, -.06), (0, 0), (0, .06)]):
        index = np.argmin(np.abs(axes[axis]-value))
        horizontal, vertical = (0, 2) if axis == 1 else (2, 1)
        remaining = [i for i in range(3) if i != axis]
        plane = np.take(display, index, axis=axis).transpose(remaining.index(vertical), remaining.index(horizontal))
        ax.imshow(plane, origin='lower', extent=[low[horizontal]-spacing[horizontal]/2, high[horizontal]+spacing[horizontal]/2,
                  low[vertical]-spacing[vertical]/2, high[vertical]+spacing[vertical]/2],
                  cmap=ListedColormap(['#ffdc78', '#64bf72', '#80aee8', '#f3f3f3', '#c167ca']), vmin=0, vmax=4, interpolation='nearest')
        origin, normal = np.zeros(3), np.zeros(3)
        origin[axis], normal[axis] = axes[axis][index], 1
        lines = trimesh.intersections.mesh_plane(tri, plane_normal=normal, plane_origin=origin)
        ax.add_collection(LineCollection(lines[:, :, [horizontal, vertical]], colors='black', linewidths=1))
        ax.set_title(f'{"XYZ"[axis]}={origin[axis]:.3f}m')
        ax.set_aspect('equal')
    fig.suptitle('Green: hair evidence | Blue: head prior | Gray: air | Yellow: unknown | Purple: conflict (unknown)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / 'evidence_sections.png', dpi=150)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build_volume_evidence_prototype()
