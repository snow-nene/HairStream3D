"""从原图/侧视头发证据生成样例头皮与开放头发表面，保留源模板。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.subject_scalp import smooth_radial_fit
from lib.template_identity import load_template_identity, sha256_file
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera


def hair_evidence(points, charts, tolerance=.002):
    positive = np.zeros(len(points), np.int32)
    veto = np.zeros(len(points), bool)
    for view, chart in charts.items():
        pixels, depth, valid = chart.pixels(points)
        x, y = pixels[valid].T
        visible = chart.head_hit[y, x] & (depth[valid] >= chart.head_z[y, x]-tolerance)
        ids = np.flatnonzero(valid)
        positive[ids[visible & chart.hair_domain[y, x]]] += 1
        if view == 'front':
            veto[ids[visible & ~chart.hair_domain[y, x]]] = True
    return positive, veto


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--views', nargs='+', default=['front', 'left', 'right', 'back'])
    p.add_argument('--smoothness', type=float, default=2000.)
    p.add_argument('--preserve-head', action='store_true', help='几何隔离对照：保留原头模，只提取头发证据')
    a = p.parse_args()
    if 'front' not in a.views:
        raise ValueError('原图 front 必须参与头发语义约束')
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须位于该图像的数据目录')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(a.head)
    with np.load(a.head) as h:
        vertices, faces = h['vertices'], h['faces']
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    source = a.data_dir/'pixal3d/hair_mesh_aligned_best.obj'
    full = o3d.io.read_triangle_mesh(str(source))
    charts = {}
    for view in a.views:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        charts[view] = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view),
                                            seg.astype(np.int32), np.zeros((*seg.shape, 2)), full)
    # 第一交点语义筛选独立于头模，避免把当前埋入头模的头发表面全部拒绝。
    full_v, full_f = np.asarray(full.vertices), np.asarray(full.triangles)
    centers = full_v[full_f].mean(1)
    positive, veto = hair_evidence(centers, charts)
    selected = (positive > 0) & ~veto
    # 用源面索引精确导出，避免纳入选中顶点之间的非头发面。
    used, remapped = np.unique(full_f[selected], return_inverse=True)
    hair = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(full_v[used]),
                                    o3d.utility.Vector3iVector(remapped.reshape(-1, 3)))
    o3d.io.write_triangle_mesh(str(a.output_dir/'hair_outer_surface.obj'), hair)
    np.savez_compressed(a.output_dir/'hair_surface_evidence.npz', source_face_ids=np.flatnonzero(selected),
                        support_counts=positive, front_veto=veto)
    center = np.array([.003, 1.765, .0])
    radius = np.linalg.norm(vertices-center, axis=1)
    radial = (vertices-center)/radius[:, None]
    rays = np.c_[vertices+.07*radial, -radial].astype(np.float32)
    cast = charts['front'].scene.cast_rays(o3d.core.Tensor(rays))
    hit = np.isfinite(cast['t_hit'].numpy())
    outer = vertices+.07*radial-np.where(hit, cast['t_hit'].numpy(), .07)[:, None]*radial
    outer_support, outer_veto = hair_evidence(outer, charts)
    front_seg = charts['front'].hair_domain
    head_front = VisibleSurfaceGrowth(charts['front'].camera, front_seg.astype(np.int32),
                                     np.zeros((*front_seg.shape, 2)), mesh)
    pixels, depth, valid = head_front.pixels(vertices)
    ids = np.flatnonzero(valid)
    x, y = pixels[valid].T
    face_pin = np.zeros(len(vertices), bool)
    face_pin[ids] = (head_front.head_hit[y, x] & (depth[valid] >= head_front.head_z[y, x]-.004)
                     & ~front_seg[y, x])
    # 固定眼鼻口、耳和颈；无直接支持的冠部顶点允许邻域平滑延拓。
    movable = (vertices[:, 1] > 1.745) & ~face_pin
    crown = np.clip((vertices[:, 1]-1.80)/.07, 0, 1)
    desired_gap = .003+.003*crown
    shift_target = ((outer-vertices)*radial).sum(1)-desired_gap
    supported = movable & hit & (outer_support > 0) & ~outer_veto & (shift_target > -.06)
    if a.preserve_head:
        fitted, shift = vertices.copy(), np.zeros(len(vertices))
    else:
        fitted, shift = smooth_radial_fit(vertices, faces, shift_target, movable, supported, center,
                                        smoothness=a.smoothness)
    fitted_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(fitted), o3d.utility.Vector3iVector(faces))
    report = {'source_template': identity, 'source_mesh_sha256': sha256_file(source), 'views': a.views,
              'surface_semantics': 'open_visible_hair_surface_not_signed_volume',
              'selected_hair_faces': int(selected.sum()), 'all_mesh_faces': len(full_f),
              'movable_vertices': int(movable.sum()), 'supported_vertices': int(supported.sum()),
              'pinned_vertices': int((~movable).sum()),
              'pinned_max_displacement_m': float(np.linalg.norm(fitted[~movable]-vertices[~movable], axis=1).max()),
              'shift_quantiles_m': np.quantile(shift[movable], [0, .1, .5, .9, 1]).tolist(),
              'watertight': fitted_mesh.is_watertight(), 'self_intersection': fitted_mesh.is_self_intersecting(),
              'bound_to_final_template': False, 'smoothness': a.smoothness,
              'preserve_head': a.preserve_head,
              'semantics': ('unchanged_source_head' if a.preserve_head else
                            'subject_scalp_prior_under_observed_hair_not_hidden_scalp_truth')}
    (a.output_dir/'fit_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report['watertight'] or report['self_intersection'] or report['pinned_max_displacement_m'] != 0:
        raise RuntimeError('头皮几何验收失败，未输出候选头模')
    np.savez_compressed(a.output_dir/'subject_head_candidate.npz', vertices=fitted, faces=faces,
                        source_vertices=vertices, radial_center=center, radial_shift=shift,
                        movable=movable, supported=supported, desired_gap=desired_gap)


if __name__ == '__main__':
    main()
