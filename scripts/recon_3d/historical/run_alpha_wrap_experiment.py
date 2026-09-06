"""三档 Alpha Wrapping 封闭代理实验，保存拓扑、轮廓和截面对比。"""
import json
import subprocess
import time
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def run_alpha_wrap_experiment():
    root = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/hair_mesh_extraction')
    out = root / 'step_02_alpha_wrap'
    out.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(root / 'step_01_multiview_surface/hair_surface_full.obj'))
    labels, counts, _ = mesh.cluster_connected_triangles()
    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(np.asarray(labels) != np.argmax(counts))
    cleaned.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out / 'cleaned_source.off'), cleaned)
    models = [('source', cleaned)]
    report = {'removed_disconnected_faces': len(mesh.triangles)-len(cleaned.triangles),
              'source_faces': len(cleaned.triangles), 'variants': {},
              'note': '仅保留最大连通主体，原始结果未覆盖。Alpha包络仅保证几何封闭，不等同于正确头发体积；残余非头发几何和新建内侧面仍需检查。'}
    source_scene = o3d.t.geometry.RaycastingScene()
    source_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cleaned))
    o3d.utility.random.seed(42)
    source_points = np.asarray(cleaned.sample_points_uniformly(30000).points).astype(np.float32)
    variants = [('fine', .003, .0005), ('medium', .006, .001), ('coarse', .012, .002)]
    if '--visualize-only' in sys.argv:
        report = json.loads((out / 'report.json').read_text())
        models.extend((name, o3d.io.read_triangle_mesh(str(out / f'{name}.off'))) for name, _, _ in variants)
        variants = []
    for name, alpha, offset in variants:
        start = time.monotonic()
        command = [str(out / 'alpha_wrap_hair'), str(out / 'cleaned_source.off'),
                   str(out / f'{name}.off'), str(alpha), str(offset)]
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        wrapped = o3d.io.read_triangle_mesh(str(out / f'{name}.off'))
        wrapped.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out / f'hair_watertight_{name}.obj'), wrapped)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wrapped))
        distance = scene.compute_distance(o3d.core.Tensor(source_points)).numpy()
        points = np.asarray(wrapped.sample_points_uniformly(30000).points).astype(np.float32)
        reverse = source_scene.compute_distance(o3d.core.Tensor(points)).numpy()
        component_ids, component_counts, _ = wrapped.cluster_connected_triangles()
        tri = trimesh.Trimesh(np.asarray(wrapped.vertices), np.asarray(wrapped.triangles), process=False)
        report['variants'][name] = {'alpha_m': alpha, 'offset_m': offset, 'elapsed_s': time.monotonic()-start,
            'vertices': len(wrapped.vertices), 'faces': len(wrapped.triangles),
            'watertight': wrapped.is_watertight(), 'self_intersection_pairs': len(wrapped.get_self_intersecting_triangles()),
            'boundary_or_nonmanifold_edges': len(wrapped.get_non_manifold_edges(allow_boundary_edges=False)),
            'components': len(component_counts), 'absolute_volume_m3': abs(float(tri.volume)),
            'source_to_wrap_q50_q95_mm': (np.quantile(distance, [.5, .95])*1000).tolist(),
            'wrap_to_source_q50_q95_mm': (np.quantile(reverse, [.5, .95])*1000).tolist()}
        print(name, report['variants'][name], flush=True)
        models.append((name, wrapped))
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # Common scale orthographic views with ray-based self-occlusion.
    center = cleaned.get_axis_aligned_bounding_box().get_center()
    radius = np.linalg.norm(np.asarray(cleaned.vertices)-center, axis=1).max()
    size = 480
    values = np.linspace(-radius*1.1, radius*1.1, size)
    xx, yy = np.meshgrid(values, -values)
    rows = []
    source_masks = {}
    for name, model in models:
        panels = []
        report.setdefault('silhouette_iou_to_cleaned', {})[name] = {}
        for yaw in [0, 90, 180, 270]:
            angle = np.deg2rad(yaw)
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            vertices = (np.asarray(model.vertices)-center)@rotation.T
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype(np.float32)),
                o3d.core.Tensor(np.asarray(model.triangles).astype(np.int32))))
            rays = np.stack([xx, yy, np.full_like(xx, 2), np.zeros_like(xx), np.zeros_like(xx), -np.ones_like(xx)], -1).astype(np.float32)
            result = scene.cast_rays(o3d.core.Tensor(rays))
            hit = np.isfinite(result['t_hit'].numpy())
            normals = result['primitive_normals'].numpy()
            shade = .3+.7*np.abs(normals@np.array([.3, .5, .8124]))
            canvas = np.full((size, size, 3), 242, np.uint8)
            canvas[hit] = np.clip(shade[hit, None]*[90, 170, 220], 0, 255).astype(np.uint8)
            cv2.putText(canvas, f'{name} | yaw {yaw}', (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1, cv2.LINE_AA)
            if name == 'source':
                source_masks[yaw] = hit
            iou = (hit & source_masks[yaw]).sum()/max(1, (hit | source_masks[yaw]).sum())
            report['silhouette_iou_to_cleaned'][name][str(yaw)] = float(iou)
            panels.append(canvas)
        row = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out / f'{name}_four_views.png'), row)
        rows.append(row)
    cv2.imwrite(str(out / 'comparison.png'), np.concatenate(rows, axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, height in zip(axes, [1.70, 1.78, 1.84]):
        for (name, model), color in zip(models, ['gray', 'green', 'royalblue', 'darkorange']):
            tri = trimesh.Trimesh(np.asarray(model.vertices), np.asarray(model.triangles), process=False)
            segments = trimesh.intersections.mesh_plane(tri, plane_origin=[0, height, 0], plane_normal=[0, 1, 0])
            if len(segments):
                ax.add_collection(LineCollection(segments[:, :, [0, 2]], colors=color, linewidths=.8, label=name))
                ax.autoscale_view()
        ax.set_aspect('equal')
        ax.set_title(f'y={height:.2f}m; X-Z cross-section')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'sections.png', dpi=150)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    run_alpha_wrap_experiment()
