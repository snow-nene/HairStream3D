"""只读诊断已保存的实际 PDE 域、头模边界与发根连通性。"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import distance_transform_edt, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


def audit_pde_volume_sections():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    run = base / 'pde_governance/current_full_reconstruction_10k_128_20260904'
    out = base / 'pde_governance/volume_domain_audit/step_01_sections'
    out.mkdir(parents=True, exist_ok=True)
    source = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    head = o3d.io.read_triangle_mesh('data/head_model.obj')
    flame = o3d.io.read_triangle_mesh(str(base / 'pde_governance/flame_head_fit/step_01_face_shape_fit/fitted_head.obj'))
    debug = np.load(run / 'fusion_debug.npz')
    sdf = np.load(run / 'pde_domain_sdf.npy')
    domain = sdf > 0
    metrics = json.loads((run / 'pde_solver_metrics.json').read_text())
    assert int(domain.sum()) == metrics['domain_voxels']
    # Reconstruct bounds using the production rule, then validate voxel spacing
    # against the saved distance field rather than silently trusting the rule.
    low = np.minimum(source.get_min_bound(), head.get_min_bound())-.03
    high = np.maximum(source.get_max_bound(), head.get_max_bound())+.03
    shape = np.array(domain.shape)
    spacing = (high-low)/(shape-1)
    reconstructed = distance_transform_edt(domain, sampling=spacing)-distance_transform_edt(~domain, sampling=spacing)
    sdf_error = float(np.max(np.abs(reconstructed-sdf)))
    if sdf_error > 2e-6:
        raise RuntimeError(f'体积坐标尺度验证失败: max sdf difference={sdf_error}')
    axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
    roots = np.load(run / 'root_view_guidance.npz')['roots_world']
    root_ids = np.rint((roots-low)/spacing).astype(int)
    in_bounds = ((root_ids >= 0) & (root_ids < shape)).all(1)
    inside = np.zeros(len(roots), bool)
    inside[in_bounds] = domain[tuple(root_ids[in_bounds].T)]
    components, count = label(domain)
    boundary = debug['boundary_mask'] & domain
    constrained = np.unique(components[boundary])
    connected = np.zeros(len(roots), bool)
    connected[inside] = np.isin(components[tuple(root_ids[inside].T)], constrained)
    nearest_distance = distance_transform_edt(~domain, sampling=spacing)
    root_distance = np.full(len(roots), np.nan)
    root_distance[in_bounds] = nearest_distance[tuple(root_ids[in_bounds].T)]
    geometry = []
    for name, mesh, color in [('Pixal3D', source, 'black'), ('standard head', head, 'royalblue'), ('fitted FLAME (uncertain)', flame, 'darkorange')]:
        geometry.append((name, trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False), color))
    fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    sections = [(1, 1.70), (1, 1.78), (1, 1.84), (0, -.06), (0, 0.), (0, .06)]
    for ax, (normal_axis, requested) in zip(axs.ravel(), sections):
        index = int(np.argmin(np.abs(axes[normal_axis]-requested)))
        value = axes[normal_axis][index]
        horizontal, vertical = (0, 2) if normal_axis == 1 else (2, 1)
        plane = np.take(domain, index, axis=normal_axis)
        original_axes = [i for i in range(3) if i != normal_axis]
        plane = np.transpose(plane, [original_axes.index(vertical), original_axes.index(horizontal)])
        extent = [low[horizontal]-spacing[horizontal]/2, high[horizontal]+spacing[horizontal]/2,
                  low[vertical]-spacing[vertical]/2, high[vertical]+spacing[vertical]/2]
        ax.imshow(plane, origin='lower', extent=extent, cmap=ListedColormap(['white', '#b7e4b0']), interpolation='nearest')
        origin = np.zeros(3)
        origin[normal_axis] = value
        normal = np.zeros(3)
        normal[normal_axis] = 1
        for name, mesh, color in geometry:
            lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
            if len(lines):
                ax.add_collection(LineCollection(lines[:, :, [horizontal, vertical]], colors=color, linewidths=1, label=name))
        nearby = np.abs(roots[:, normal_axis]-value) < spacing[normal_axis]*1.5
        for mask, color, name in [(nearby & inside, 'green', 'roots in domain'), (nearby & ~inside, 'red', 'roots outside domain')]:
            ax.scatter(roots[mask, horizontal], roots[mask, vertical], s=8, c=color, label=name)
        ax.set_title(f'{"XYZ"[normal_axis]}={value:.3f}m | green fill: actual PDE domain')
        ax.set_xlabel('XYZ'[horizontal]+' (m)')
        ax.set_ylabel('XYZ'[vertical]+' (m)')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'domain_sections.png', dpi=160)
    plt.close(fig)
    # Signed-distance indicators are explicitly uncertain because neither head is watertight.
    indices = np.argwhere(domain)[::8]
    points = (low+indices*spacing).astype(np.float32)
    suspect = {}
    for name, mesh in [('standard_head', head), ('fitted_flame', flame)]:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        distance = scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=5).numpy()
        suspect[name] = {'watertight': mesh.is_watertight(), 'sampled_domain_points': len(points),
                         'signed_inside_beyond_3mm_fraction_UNCERTAIN': float((distance < -.003).mean())}
    report = {'source_run': str(run), 'domain_voxels': int(domain.sum()), 'shape': shape.tolist(),
              'initial_hair_volume_voxels': int(debug['hair_volume'].sum()),
              'bounds_min': low.tolist(), 'bounds_max': high.tolist(), 'spacing_mm': (spacing*1000).tolist(),
              'bounds_provenance': '按当前生产代码 head+完整mesh包围盒±3cm重建；保存SDF验证尺度。无原始bbox元数据，绝对平移仍依赖该规则。',
              'sdf_reconstruction_max_error_m': sdf_error,
              'domain_components_6_connected': count, 'roots': len(roots),
              'roots_in_bounds': int(in_bounds.sum()), 'roots_in_domain_nearest_voxel': int(inside.sum()),
              'roots_connected_to_observed_boundary_6_connected': int(connected.sum()),
              'root_distance_to_domain_q50_q90_mm': (np.nanquantile(root_distance, [.5, .9])*1000).tolist(),
              'head_overlap_indicators': suspect,
              'note': '白色只表示不在当前PDE域，不等于头发缺失。头模内部指标因开口/自交不可靠。发根采用原运行保存值，未迁移到新拟合FLAME；体素连通不保证方向场轨迹可达。未修改PDE、参数或模型。'}
    np.savez_compressed(out / 'root_domain_audit.npz', roots=roots, inside=inside, connected=connected, distance_m=root_distance)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    audit_pde_volume_sections()
