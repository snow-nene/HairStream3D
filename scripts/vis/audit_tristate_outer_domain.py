"""外包络三态诊断：未知不是有效域，保护点不覆盖几何冲突。"""
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


def audit_tristate_outer_domain():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    parent = base / 'pde_governance/volume_domain_audit'
    out = parent / 'step_03_tristate_geometry'
    out.mkdir(parents=True, exist_ok=True)
    previous = np.load(parent / 'step_02_outer_envelope/candidate_domains.npz')
    domain = previous['baseline']
    low, high = previous['b_min'], previous['b_max']
    shape = np.array(domain.shape)
    spacing = (high-low)/(shape-1)
    axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
    run = base / 'pde_governance/current_full_reconstruction_10k_128_20260904'
    boundary = np.load(run / 'fusion_debug.npz')['boundary_mask'] & domain
    roots = np.load(run / 'root_view_guidance.npz')['roots_world']
    root_ids = np.rint((roots-low)/spacing).astype(int)
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    votes = np.zeros(shape, np.uint8)
    brackets = np.zeros(shape, np.uint8)
    missing_pairs = np.zeros(shape, np.uint8)
    margin = .006
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        aa, bb = np.meshgrid(axes[others[0]], axes[others[1]], indexing='ij')
        surfaces = []
        for sign in [1, -1]:
            start = low[axis]-.1 if sign == 1 else high[axis]+.1
            origins = np.zeros((aa.size, 3), np.float32)
            origins[:, axis] = start
            origins[:, others[0]], origins[:, others[1]] = aa.ravel(), bb.ravel()
            direction = np.zeros_like(origins)
            direction[:, axis] = sign
            hit = scene.cast_rays(o3d.core.Tensor(np.column_stack([origins, direction])))['t_hit'].numpy()
            reshape = [shape[i] if i != axis else 1 for i in range(3)]
            surfaces.append((start+sign*hit).reshape(reshape))
        lower, upper = surfaces
        valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper+1e-6)
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        coordinates = axes[axis].reshape(reshape)
        brackets += valid & (coordinates >= lower-margin) & (coordinates <= upper+margin)
        # Each axis contributes at most one outside vote.
        votes += (np.isfinite(lower) & (coordinates < lower-margin)) | (np.isfinite(upper) & (coordinates > upper+margin))
        missing_pairs += ~valid
    supported = domain & (brackets == 3) & (votes == 0)
    outside = domain & (votes >= 2)
    uncertain = domain & ~supported & ~outside
    assert not (supported & outside).any()
    assert np.array_equal(supported | outside | uncertain, domain)
    classes = np.zeros(shape, np.uint8)
    classes[supported], classes[outside], classes[uncertain] = 1, 2, 3
    names = ['outside_original_domain', 'three_axis_envelope_supported', 'multi_axis_exterior', 'uncertain']
    root_class = classes[tuple(root_ids.T)]
    protected = previous['protection']
    old_keep = previous['margin6mm_votes2']
    components, component_count = label(supported)
    constrained = np.unique(components[boundary & supported])
    constrained = constrained[constrained != 0]
    root_components = components[tuple(root_ids.T)]
    connected = (root_components != 0) & np.isin(root_components, constrained)
    report = {'margin_m': margin, 'baseline_voxels': int(domain.sum()),
              'class_counts': {names[i]: int((classes == i).sum()) for i in [1, 2, 3]},
              'root_counts': {names[i]: int((root_class == i).sum()) for i in range(4)},
              'boundary_counts': {names[i]: int((boundary & (classes == i)).sum()) for i in [1, 2, 3]},
              'old_kept_now_exterior': int((old_keep & outside).sum()),
              'old_kept_now_uncertain': int((old_keep & uncertain).sum()),
              'protected_exterior_voxels': int((protected & outside).sum()),
              'protected_uncertain_voxels': int((protected & uncertain).sum()),
              'uncertain_with_missing_ray_pairs': int((uncertain & (missing_pairs > 0)).sum()),
              'supported_components_6_connected': component_count,
              'supported_roots_connected_to_boundary': int(connected.sum()),
              'production_ready': False,
              'note': '绿色仅表示三个轴向包络均支持，不证明真实头发内部；仍包含头部和凹陷歧义。黄色未知不进入候选有效域；红色多轴外部证据。保护点不强制恢复，冲突单列。未重求PDE或修改正式管线。'}
    np.savez_compressed(out / 'tristate_domain.npz', classes=classes, outside_votes=votes,
                        bracket_axes=brackets, missing_pairs=missing_pairs, b_min=low, b_max=high,
                        root_classes=root_class, boundary=boundary)
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    head = o3d.io.read_triangle_mesh('data/head_model.obj')
    geometries = [(trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False), color, name)
                  for m, color, name in [(mesh, 'black', 'Pixal3D'), (head, 'royalblue', 'standard head')]]
    for ax, (axis, target) in zip(axs.ravel(), [(1, 1.70), (1, 1.78), (1, 1.84), (0, -.06), (0, 0), (0, .06)]):
        index = np.argmin(np.abs(axes[axis]-target))
        horizontal, vertical = (0, 2) if axis == 1 else (2, 1)
        remaining = [i for i in range(3) if i != axis]
        plane = np.take(classes, index, axis=axis).transpose(remaining.index(vertical), remaining.index(horizontal))
        ax.imshow(plane, origin='lower', extent=[low[horizontal]-spacing[horizontal]/2, high[horizontal]+spacing[horizontal]/2,
                  low[vertical]-spacing[vertical]/2, high[vertical]+spacing[vertical]/2],
                  cmap=ListedColormap(['white', '#b7e4b0', '#f4a29a', '#ffdc78']), vmin=0, vmax=3, interpolation='nearest')
        origin, normal = np.zeros(3), np.zeros(3)
        origin[axis], normal[axis] = axes[axis][index], 1
        for tri, color, name in geometries:
            lines = trimesh.intersections.mesh_plane(tri, plane_normal=normal, plane_origin=origin)
            ax.add_collection(LineCollection(lines[:, :, [horizontal, vertical]], colors=color, linewidths=1, label=name))
        ax.set_title(f'{"XYZ"[axis]}={origin[axis]:.3f}m')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
    fig.suptitle('Green: envelope-supported (NOT validated hair) | Red: exterior evidence | Yellow: uncertain', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / 'tristate_sections.png', dpi=150)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    audit_tristate_outer_domain()
