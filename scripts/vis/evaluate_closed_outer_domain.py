"""以完整模型封闭包络替代漏射外部判定；只生成独立候选域。"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


def evaluate_closed_outer_domain():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    parent = base / 'pde_governance/volume_domain_audit'
    input_dir = parent / 'step_04_root_ray_consistency'
    out = parent / 'step_05_closed_external_envelope'
    out.mkdir(parents=True, exist_ok=True)
    prior = np.load(parent / 'step_03_tristate_geometry/tristate_domain.npz')
    domain = prior['classes'] > 0
    low, high = prior['b_min'], prior['b_max']
    spacing = (high-low)/(np.array(domain.shape)-1)
    axes = [np.linspace(low[i], high[i], domain.shape[i]) for i in range(3)]
    wrap = o3d.io.read_triangle_mesh(str(input_dir / 'full_model_outer_wrap_large.off'))
    if not wrap.is_watertight():
        raise RuntimeError('完整模型包络不水密，拒绝内外判断')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wrap))
    ids = np.argwhere(domain)
    points = (low+ids*spacing).astype(np.float32)
    distance = scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=5).numpy()
    candidate = np.zeros_like(domain)
    candidate[tuple(ids[distance <= .003].T)] = True
    run = base / 'pde_governance/current_full_reconstruction_10k_128_20260904'
    boundary = np.load(run / 'fusion_debug.npz')['boundary_mask'] & domain
    roots = np.load(run / 'root_view_guidance.npz')['roots_world'].astype(np.float32)
    root_distance = scene.compute_signed_distance(o3d.core.Tensor(roots), nsamples=5).numpy()
    root_ids = np.rint((roots-low)/spacing).astype(int)
    labels, components = label(candidate)
    observed_components = np.unique(labels[boundary & candidate])
    observed_components = observed_components[observed_components != 0]
    root_labels = labels[tuple(root_ids.T)]
    connected = (root_labels != 0) & np.isin(root_labels, observed_components)
    source = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    tri = trimesh.Trimesh(np.asarray(wrap.vertices), np.asarray(wrap.triangles), process=False)
    boundary_kept = int((boundary & candidate).sum())
    report = {'watertight': True, 'self_intersection_pairs': len(wrap.get_self_intersecting_triangles()),
              'wrap_alpha_m': .10, 'wrap_offset_m': .001, 'outer_tolerance_m': .003,
              'wrap_faces': len(wrap.triangles), 'wrap_volume_m3': abs(float(tri.volume)),
              'baseline_voxels': int(domain.sum()), 'candidate_voxels': int(candidate.sum()),
              'removed_fraction': float((domain & ~candidate).sum()/domain.sum()),
              'boundary_voxels': int(boundary.sum()), 'boundary_kept': boundary_kept,
              'baseline_roots_in_domain': int(domain[tuple(root_ids.T)].sum()),
              'candidate_roots_in_domain': int(candidate[tuple(root_ids.T)].sum()),
              'roots_within_wrap_plus_tolerance_exact': int((root_distance <= .003).sum()),
              'old_1208_conflicts_inside_wrap_exact': int(((prior['root_classes'] == 2) & (root_distance <= .003)).sum()),
              'components': components, 'roots_connected_to_boundary': int(connected.sum()),
              'production_ready': False,
              'note': '封闭完整头脸外形，不是头发薄壳；不修改头模和发根。包络会填补凹陷并有几何偏差，仅验证外边界策略，尚未扣除头部、处理分区或重求PDE。'}
    np.savez_compressed(out / 'closed_outer_candidate.npz', domain=candidate, b_min=low, b_max=high,
                        root_signed_distance=root_distance, boundary_kept=boundary & candidate)
    geometries = [(trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False), color, name)
                  for m, color, name in [(source, 'black', 'original mesh'), (wrap, 'purple', 'closed outer envelope')]]
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    colors = candidate.astype(np.uint8)
    colors[domain & ~candidate] = 2
    for ax, (axis, target) in zip(axs.ravel(), [(1, 1.70), (1, 1.78), (1, 1.84), (0, -.06), (0, 0), (0, .06)]):
        index = np.argmin(np.abs(axes[axis]-target))
        horizontal, vertical = (0, 2) if axis == 1 else (2, 1)
        remaining = [i for i in range(3) if i != axis]
        plane = np.take(colors, index, axis=axis).transpose(remaining.index(vertical), remaining.index(horizontal))
        ax.imshow(plane, origin='lower', extent=[low[horizontal]-spacing[horizontal]/2, high[horizontal]+spacing[horizontal]/2,
                  low[vertical]-spacing[vertical]/2, high[vertical]+spacing[vertical]/2],
                  cmap=ListedColormap(['white', '#b7e4b0', '#f4a29a']), vmin=0, vmax=2, interpolation='nearest')
        origin, normal = np.zeros(3), np.zeros(3)
        origin[axis], normal[axis] = axes[axis][index], 1
        for mesh, color, name in geometries:
            lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
            ax.add_collection(LineCollection(lines[:, :, [horizontal, vertical]], colors=color, linewidths=1, label=name))
        ax.set_aspect('equal')
        ax.set_title(f'{"XYZ"[axis]}={origin[axis]:.3f}m | green retained / red removed')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'closed_outer_sections.png', dpi=150)
    plt.close(fig)
    (out / 'closed_outer_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    evaluate_closed_outer_domain()
