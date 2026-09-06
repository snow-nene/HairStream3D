"""精确发根与九射线邻域复核；不将不稳定外部投票当作确定穿出。"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def resolve_root_envelope_conflicts():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    parent = base / 'pde_governance/volume_domain_audit'
    out = parent / 'step_04_root_ray_consistency'
    out.mkdir(parents=True, exist_ok=True)
    prior = np.load(parent / 'step_03_tristate_geometry/tristate_domain.npz')
    run = base / 'pde_governance/current_full_reconstruction_10k_128_20260904'
    roots = np.load(run / 'root_view_guidance.npz')['roots_world'].astype(np.float32)
    old = prior['root_classes']
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(roots)).numpy()
    offsets = np.array([(a, b) for a in [-.003, 0, .003] for b in [-.003, 0, .003]])
    deviation = np.full((len(roots), 6, 9), np.nan)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign_index, sign in enumerate([1, -1]):
            origin = np.repeat(roots[:, None, :], 9, axis=1)
            origin[:, :, others[0]] += offsets[None, :, 0]
            origin[:, :, others[1]] += offsets[None, :, 1]
            start = prior['b_min'][axis]-.1 if sign == 1 else prior['b_max'][axis]+.1
            origin[:, :, axis] = start
            directions = np.zeros_like(origin)
            directions[:, :, axis] = sign
            hits = scene.cast_rays(o3d.core.Tensor(np.concatenate([origin, directions], -1).reshape(-1, 6)))['t_hit'].numpy().reshape(-1, 9)
            surface = start+sign*hits
            values = sign*(surface-roots[:, axis, None])
            deviation[:, 2*axis+sign_index] = np.where(np.isfinite(hits), values, np.nan)
    # Root-only diagnostic keeps the prior 6mm tolerance; neighbor rays give
    # sensitivity evidence, not proof that a ray crossed a hole.
    exact_out = np.any((deviation[:, :, 4] > .006).reshape(-1, 3, 2), axis=2).sum(1)
    valid = np.isfinite(deviation)
    strong_out = ((deviation > .006).sum(2) >= 7)
    strong_inside = ((deviation <= .006).sum(2) >= 7)
    robust_out = strong_out.reshape(-1, 3, 2).any(2).sum(1)
    robust_support = strong_inside.all(1)
    status = np.full(len(roots), 3, np.uint8)
    status[robust_support] = 1
    status[robust_out >= 2] = 2
    status[old == 0] = 0
    original_conflict = old == 2
    unstable = original_conflict & (robust_out < 2)
    stable_conflict = original_conflict & (robust_out >= 2)
    # Two millimeters from the observed surface is a useful severity indicator,
    # but unsigned distance cannot distinguish a root inside from outside.
    groups = {'all_roots': np.ones(len(roots), bool), 'previous_conflicts': original_conflict,
              'persistent_neighborhood_conflicts': stable_conflict,
              'ray_sensitive_previous_conflicts': unstable}
    report = {'roots': len(roots), 'previous_conflict_count': int(original_conflict.sum()),
              'previous_conflicts_not_reproduced_at_exact_coordinates': int((original_conflict & (exact_out < 2)).sum()),
              'previous_conflicts_sensitive_to_neighbor_rays': int(unstable.sum()),
              'previous_conflicts_persistent_in_neighbor_rays': int(stable_conflict.sum()),
              'revised_status_counts': {str(i): int((status == i).sum()) for i in range(4)},
              'distance_statistics': {}, 'production_ready': False,
              'note': '0原域外，1六方向均至少7/9邻域射线支持包络，2至少两轴有7/9外部证据，3未知。只复核发根，未改发根坐标或PDE。邻域敏感可能来自孔洞、真实沟槽或薄表面，不能自动当作孔洞修补。持续冲突也不是已证实穿头。'}
    for name, mask in groups.items():
        report['distance_statistics'][name] = {'count': int(mask.sum()),
            'unsigned_mesh_distance_q50_q90_mm': (np.quantile(distances[mask], [.5, .9])*1000).tolist() if mask.any() else [],
            'within_2mm': int((mask & (distances <= .002)).sum())}
    np.savez_compressed(out / 'root_evidence.npz', roots=roots, old_status=old, revised_status=status,
                        deviation_m=deviation, mesh_distance_m=distances, exact_outside_axes=exact_out)
    vertices = np.asarray(mesh.vertices)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (x, y), title in zip(axs, [(0, 1), (2, 1), (0, 2)], ['Front', 'Side', 'Top']):
        ax.scatter(vertices[::40, x], vertices[::40, y], s=.2, c='gray', alpha=.25)
        ax.scatter(roots[~original_conflict, x], roots[~original_conflict, y], s=1, c='silver', alpha=.2)
        for mask, color, title2 in [(unstable, 'darkorange', 'ray-sensitive conflict'), (stable_conflict, 'red', 'persistent conflict')]:
            ax.scatter(roots[mask, x], roots[mask, y], s=4, c=color, label=title2)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle('Prior 1,208 conflict roots: exact-coordinate and 3mm-neighborhood audit')
    fig.tight_layout()
    fig.savefig(out / 'conflict_roots.png', dpi=160)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    resolve_root_envelope_conflicts()
