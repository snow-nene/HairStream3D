"""六向首交点外包络裁剪实验；不依赖非水密模型的有符号内外判断。"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_dilation, distance_transform_edt, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


def experiment_pde_outer_envelope():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    run = base / 'pde_governance/current_full_reconstruction_10k_128_20260904'
    out = base / 'pde_governance/volume_domain_audit/step_02_outer_envelope'
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((base / 'pde_governance/volume_domain_audit/step_01_sections/report.json').read_text())
    low, high = np.array(audit['bounds_min']), np.array(audit['bounds_max'])
    domain = np.load(run / 'pde_domain_sdf.npy') > 0
    debug = np.load(run / 'fusion_debug.npz')
    boundary = debug['boundary_mask'] & domain
    shape = np.array(domain.shape)
    spacing = (high-low)/(shape-1)
    axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
    roots = np.load(run / 'root_view_guidance.npz')['roots_world']
    root_ids = np.rint((roots-low)/spacing).astype(int)
    assert ((root_ids >= 0) & (root_ids < shape)).all()
    baseline_roots = domain[tuple(root_ids.T)]
    protection = boundary.copy()
    protection[tuple(root_ids[baseline_roots].T)] = True
    protection = binary_dilation(protection, structure=np.ones((3, 3, 3), bool)) & domain
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    deviations = []
    # Missing ray intersections are unknown, never automatic background votes.
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        aa, bb = np.meshgrid(axes[others[0]], axes[others[1]], indexing='ij')
        for sign in [1, -1]:
            start = low[axis]-.1 if sign == 1 else high[axis]+.1
            origins = np.zeros((aa.size, 3), np.float32)
            origins[:, axis] = start
            origins[:, others[0]], origins[:, others[1]] = aa.ravel(), bb.ravel()
            directions = np.zeros_like(origins)
            directions[:, axis] = sign
            hit = scene.cast_rays(o3d.core.Tensor(np.column_stack([origins, directions])))['t_hit'].numpy()
            surface = start+sign*hit
            reshape = [shape[i] if i != axis else 1 for i in range(3)]
            surface = surface.reshape(aa.shape).reshape(reshape)
            coord_shape = [1, 1, 1]
            coord_shape[axis] = shape[axis]
            coordinate = axes[axis].reshape(coord_shape)
            deviation = sign*(surface-coordinate)
            deviations.append(np.where(np.isfinite(surface), deviation, -np.inf).astype(np.float32))
    selected_indices = np.argwhere(domain)
    points = (low+selected_indices*spacing).astype(np.float32)
    distance = scene.compute_distance(o3d.core.Tensor(points)).numpy()
    surface_guard = np.zeros_like(domain)
    surface_guard[tuple(selected_indices[distance <= np.linalg.norm(spacing)].T)] = True
    # Evaluation subsets are fixed from the baseline, never reselected per variant.
    near_surface = surface_guard.copy()
    tip = near_surface.copy()
    tip[:, axes[1] > 1.70, :] = False
    variants = {}
    masks = {}
    for margin in [.003, .006, .010]:
        votes = np.zeros(shape, np.uint8)
        for deviation in deviations:
            votes += deviation > margin
        for required in [1, 2]:
            name = f'margin{round(margin*1000)}mm_votes{required}'
            raw_remove = domain & (votes >= required)
            candidate = domain & (~raw_remove | protection | surface_guard)
            assert not (candidate & ~domain).any()
            assert (candidate[boundary]).all()
            assert np.array_equal(candidate[tuple(root_ids.T)], baseline_roots)
            labels, components = label(candidate)
            boundary_components = np.unique(labels[boundary])
            root_components = labels[tuple(root_ids[baseline_roots].T)]
            anchored = np.union1d(boundary_components, root_components)
            anchored = anchored[anchored != 0]
            orphan = candidate & ~np.isin(labels, anchored)
            candidate &= ~orphan
            labels, components = label(candidate)
            boundary_components = np.unique(labels[boundary])
            root_components = labels[tuple(root_ids[baseline_roots].T)]
            connected = np.isin(root_components, boundary_components)
            row = {'margin_mm': margin*1000, 'required_outside_votes': required,
                   'voxels': int(candidate.sum()), 'removed_voxels': int((domain & ~candidate).sum()),
                   'removed_fraction': float((domain & ~candidate).sum()/domain.sum()),
                   'root_count_in_domain': int(baseline_roots.sum()),
                   'roots_connected_to_boundary': int(connected.sum()),
                   'boundary_retention': float(candidate[boundary].mean()),
                   'near_surface_retention': float(candidate[near_surface].mean()),
                   'tip_near_surface_retention': float(candidate[tip].mean()),
                   'protected_outside_vote_voxels': int((raw_remove & candidate).sum()),
                   'components_6_connected': components,
                   'unanchored_island_voxels_removed': int(orphan.sum()),
                   'gate_passed': bool(connected.all() and components == 1 and (domain & ~candidate).any())}
            variants[name], masks[name] = row, candidate
            print(name, row, flush=True)
    # Conservative two-direction agreement; 6mm is about two baseline voxels.
    preferred = 'margin6mm_votes2'
    if not variants[preferred]['gate_passed']:
        raise RuntimeError('保守候选未通过发根/连通检查，不输出推荐域')
    candidate = masks[preferred]
    np.savez_compressed(out / 'candidate_domains.npz', **masks, baseline=domain,
                        protection=protection, surface_guard=surface_guard, b_min=low, b_max=high)
    np.save(out / 'recommended_domain.npy', candidate)
    signed = distance_transform_edt(candidate, sampling=spacing)-distance_transform_edt(~candidate, sampling=spacing)
    np.save(out / 'recommended_domain_sdf.npy', signed.astype(np.float32))
    tri = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    head = o3d.io.read_triangle_mesh('data/head_model.obj')
    head_tri = trimesh.Trimesh(np.asarray(head.vertices), np.asarray(head.triangles), process=False)
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (axis, target) in zip(axs.ravel(), [(1, 1.70), (1, 1.78), (1, 1.84), (0, -.06), (0, 0), (0, .06)]):
        index = np.argmin(np.abs(axes[axis]-target))
        horizontal, vertical = (0, 2) if axis == 1 else (2, 1)
        colors = np.zeros(domain.shape, np.uint8)
        colors[candidate] = 1
        colors[domain & ~candidate] = 2
        plane = np.take(colors, index, axis=axis)
        remaining = [i for i in range(3) if i != axis]
        plane = plane.transpose(remaining.index(vertical), remaining.index(horizontal))
        ax.imshow(plane, origin='lower', extent=[low[horizontal]-spacing[horizontal]/2, high[horizontal]+spacing[horizontal]/2,
                  low[vertical]-spacing[vertical]/2, high[vertical]+spacing[vertical]/2],
                  cmap=ListedColormap(['white', '#b7e4b0', '#f4a29a']), vmin=0, vmax=2, interpolation='nearest')
        origin, normal = np.zeros(3), np.zeros(3)
        origin[axis], normal[axis] = axes[axis][index], 1
        for model, color, name in [(tri, 'black', 'Pixal3D'), (head_tri, 'royalblue', 'unchanged head')]:
            lines = trimesh.intersections.mesh_plane(model, plane_normal=normal, plane_origin=origin)
            ax.add_collection(LineCollection(lines[:, :, [horizontal, vertical]], colors=color, linewidths=1, label=name))
        ax.set_title(f'{"XYZ"[axis]}={origin[axis]:.3f}m | green kept / red removed')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'trim_sections.png', dpi=150)
    plt.close(fig)
    report = {'baseline_run': str(run), 'baseline_voxels': int(domain.sum()),
              'recommended': preferred, 'variants': variants,
              'note': '只裁剪域，不重求PDE或生成发丝。六轴首交点外侧投票不是精确内外分类；漏射保留。保留原发根一圈和观测边界一圈以及距原模型一个体素对角线内区域。表面和发根覆盖是显式保护结果，不是独立精度验证。未解决头内误包含、分区或内部发流问题。'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    experiment_pde_outer_envelope()
