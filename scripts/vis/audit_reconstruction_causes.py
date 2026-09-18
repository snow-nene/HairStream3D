"""固定发丝与原图，诊断头模替换、外包络、方向抵消和长度截断。"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.template_identity import load_template_identity, require_bound_template, sha256_file
from scripts.recon_3d.grow_visible_surface_gaps import (
    LocalGuideField, VisibleSurfaceGrowth, load_observation_camera,
)


def quantiles(values):
    values = np.asarray(values)
    return np.quantile(values, [0, .1, .5, .9, .99, 1]).tolist() if values.size else None


def scene_for(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def distances(scene, points, signed=False):
    method = scene.compute_signed_distance if signed else scene.compute_distance
    return method(o3d.core.Tensor(np.asarray(points, np.float32)),
                  **({'nsamples': 11} if signed else {})).numpy()


def conflict_points(points, front):
    ids, depth, valid = front.pixels(points)
    conflict = np.zeros(len(points), bool)
    outside_distance = np.full(len(points), np.nan)
    x, y = ids[valid].T
    outside = distance_transform_edt(~front.hair_domain)
    outside_distance[valid] = outside[y, x]
    conflict[valid] = (front.head_hit[y, x] & (depth[valid] >= front.head_z[y, x] - 1e-5)
                       & ~front.hair_domain[y, x])
    return conflict, outside_distance


def write_categories(path, shape, masks_and_colors):
    image = np.zeros((*shape, 3), np.uint8)
    for mask, color in masks_and_colors:
        image[mask] = color
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def interpolation_diagnostic(values, weights, normal):
    """区分保存场本身弱、邻域抵消和最后切向投影损失。"""
    weights = np.asarray(weights, float) / np.sum(weights)
    before = np.sum(values * weights[:, None], axis=0)
    after = before - normal * (before @ normal)
    norms = np.linalg.norm(values, axis=1)
    input_norm = float(weights @ norms)
    if not np.any(norms > 1e-10):
        category = 'outside_saved_support'
    elif np.linalg.norm(before) >= .05 and np.linalg.norm(after) < .05:
        category = 'tangent_projection_loss'
    elif input_norm >= .05 and np.linalg.norm(before) < .05:
        category = 'saved_field_direction_cancellation'
    elif np.linalg.norm(before) < .05:
        category = 'weak_saved_field'
    else:
        category = 'not_below_query_threshold'
    return {'category': category, 'nearest_saved_vectors_norm': norms.tolist(),
            'weighted_input_norm': input_norm,
            'interpolation_coherence': float(np.linalg.norm(before) / max(input_norm, 1e-12)),
            'before_projection_norm': float(np.linalg.norm(before)),
            'after_projection_norm': float(np.linalg.norm(after))}


def replay_unresolved(args):
    """重放方向未解析候选的积分停止点；不导出或替换任何重建发丝。"""
    from lib.surface_guide_field import SurfaceGuideField
    from lib.supplemental_growth import execute_supplemental_growth
    from scripts.recon_3d.grow_visible_surface_gaps import guard_visible_photo_domain

    growth = json.loads((args.run_dir / 'report.json').read_text())
    identity = load_template_identity(args.head)
    if growth.get('coverage_camera') is None or growth.get('visibility_render') not in (None, 'None'):
        raise ValueError('停止点重放仅支持没有模板图像守卫的固定目标实验')
    with np.load(args.head) as d:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(d['vertices']),
                                       o3d.utility.Vector3iVector(d['faces']))
    with np.load(args.run_dir / 'view_evidence.npz') as d:
        camera, labels = d['camera'], d['labels']
        selected_ids, roots = d['selected_candidate_indices'], d['selected_roots']
    seg = cv2.imread(str(args.data_dir / 'maps/seg' / f"{growth['view']}.png"), 0) > 127
    chart = VisibleSurfaceGrowth(camera, labels, np.zeros((*seg.shape, 2)), mesh,
                                 clearance=growth['surface_offset_m'], hair_domain=seg,
                                 contour_correction=growth['semantic_direction_correction'])
    front_seg = cv2.imread(str(args.data_dir / 'maps/seg/front.png'), 0) > 127
    front = VisibleSurfaceGrowth(load_observation_camera(args.data_dir, 'front'),
                                front_seg.astype(np.int32), np.zeros((*front_seg.shape, 2)), mesh)
    field = SurfaceGuideField.__new__(SurfaceGuideField)
    field.chart, field.sign, field.branch_counts = chart, 1., {'forward': 0, 'reverse': 0}
    with np.load(args.run_dir / 'surface_direction_field.npz') as d:
        field.vertices, field.normals = d['vertices'], d['normals']
        field.fields = {int(k.split('_')[1]): d[k] for k in d.files if k.startswith('region_')}
    field.tree = cKDTree(field.vertices)
    lookup = {int(i): p for i, p in zip(selected_ids, roots)}
    expected = [r for r in growth['records'] if r['stop_reason'] == 'unresolved_surface_direction']
    evidence = []

    def guard(start, end, partition):
        return guard_visible_photo_domain(start, end, front) or chart.guard(start, end, partition)

    def begin(root, partition):
        if growth['surface_branch_selection']:
            field.begin_trajectory(root, partition, guard)
        evidence.append({'branch_sign': field.sign})

    def query(point, partition):
        vector, reason = field.query(point, partition)
        if reason == 'unresolved_surface_direction':
            distance, ids = field.tree.query(point, k=4)
            weights = 1 / np.maximum(distance, .001) ** 2
            values = field.fields[partition][ids]
            nearest = chart.scene.compute_closest_points(o3d.core.Tensor(point[None].astype(np.float32)))
            normal = nearest['primitive_normals'].numpy()[0]
            evidence[-1].update({'point_m': point.tolist(), **interpolation_diagnostic(values, weights, normal)})
        return vector, reason

    plan = {'roots': {'indices': [r['candidate_id'] for r in expected],
                      'points': [lookup[r['candidate_id']] for r in expected],
                      'partitions': [r['partition'] for r in expected]}}
    result = execute_supplemental_growth(plan, query_field=query, check_segment=guard,
                                         audit_trajectory=lambda _: {'passed': True}, coverage_score=lambda _: 0.,
                                         existing_strands=[], step_m=.0005, length_m=.12, min_length_m=.015,
                                         begin_trajectory=begin)
    matches = []
    for old, new, details in zip(expected, result['records'], evidence):
        matches.append(old['stop_reason'] == new['stop_reason'] and abs(old['length_m'] - new['length_m']) < 1e-8)
        details.update({'candidate_id': old['candidate_id'], 'expected_length_m': old['length_m'],
                        'replay_length_m': new['length_m'], 'replay_stop_reason': new['stop_reason'],
                        'stop_reproduced': matches[-1]})
    report = {'template_identity': identity, 'candidates': len(expected), 'matched_stop_and_length': sum(matches),
              'categories_matched_only': dict(Counter(d.get('category', 'unclassified') for d in evidence if d['stop_reproduced'])),
              'records': evidence,
              'limits': '只重放场查询与原有积分守卫；不重复轨迹质量与碰撞验收，不产生新重建。'}
    points = np.array([d['point_m'] for d in evidence if d['stop_reproduced'] and 'point_m' in d])
    if len(points):
        from scipy.sparse.csgraph import connected_components
        graph = cKDTree(points).sparse_distance_matrix(cKDTree(points), .002)
        count, clusters = connected_components(graph, directed=False)
        report['stop_clusters_2mm'] = sorted(
            [{'count': int((clusters == i).sum()), 'center_m': points[clusters == i].mean(0).tolist()}
             for i in range(count)], key=lambda x: -x['count'])
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--reference-head', type=Path, default=Path('data/head_model.obj'))
    parser.add_argument('--baseline-count', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replay-unresolved', action='store_true')
    args = parser.parse_args()
    expected_root = (args.data_dir / 'pde_governance').resolve()
    if not args.output_dir.resolve().is_relative_to(expected_root):
        raise ValueError('诊断输出必须位于该 Image ID 的 pde_governance 内')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.replay_unresolved:
        replay_unresolved(args)
        return
    source = args.run_dir / 'all_root_prefixes.npz'
    identity = load_template_identity(args.head)
    with np.load(source) as data:
        require_bound_template(data, Path(identity['template_path']))
        dense = data['strands']
        strands = [p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 0]].copy()
                   for p in dense]
        del dense
    if not 0 < args.baseline_count < len(strands):
        raise ValueError('baseline count must identify a nonempty proper prefix')
    with np.load(args.head) as data:
        head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),
                                       o3d.utility.Vector3iVector(data['faces']))
    reference_head = o3d.io.read_triangle_mesh(str(args.reference_head))
    heads = {'template': head, 'reference': reference_head}
    head_scenes = {k: scene_for(m) for k, m in heads.items()}
    closed = {k: m.is_watertight() for k, m in heads.items()}
    if not closed['template']:
        raise ValueError('绑定模板头模必须封闭')
    hair_path = args.data_dir / 'pixal3d/hair_mesh_aligned_best.obj'
    hair = o3d.io.read_triangle_mesh(str(hair_path))
    sdf_path = args.data_dir / 'pixal3d/hair_mesh_sdf.obj'
    sdf_mesh = o3d.io.read_triangle_mesh(str(sdf_path))
    report = {'template_identity': identity,
              'quantile_probabilities': [0, .1, .5, .9, .99, 1],
              'scope': 'fixed_v27_diagnostics_only_no_reconstruction_change',
              'input_sha256': {str(p.resolve()): sha256_file(p) for p in
                               [source, args.head, args.reference_head, hair_path, sdf_path,
                                args.data_dir / 'maps/param/front.npy']},
              'hair_sdf_same_geometry': bool(np.array_equal(np.asarray(hair.vertices), np.asarray(sdf_mesh.vertices))
                                             and np.array_equal(np.asarray(hair.triangles), np.asarray(sdf_mesh.triangles)))}
    del sdf_mesh
    welded = copy.deepcopy(hair)
    welded.merge_close_vertices(1e-6)
    welded.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    _, counts, areas = welded.cluster_connected_triangles()
    faces = np.asarray(welded.triangles)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, multiplicity = np.unique(edges, axis=0, return_counts=True)
    bad_midpoints = np.asarray(welded.vertices)[edges[multiplicity > 2]].mean(1)
    bad_tree = cKDTree(bad_midpoints) if len(bad_midpoints) else None
    report['hair_topology'] = {'weld_tolerance_m': 1e-6, 'watertight': welded.is_watertight(),
                               'components': len(counts), 'largest_area_fraction': max(areas) / sum(areas),
                               'boundary_edges': int((multiplicity == 1).sum()),
                               'nonmanifold_edges': int((multiplicity > 2).sum())}
    print(json.dumps({'stage': 'topology', 'result': report['hair_topology']}), flush=True)
    report['heads'] = {k: {'watertight': closed[k], 'bounds_m': [np.asarray(m.vertices).min(0).tolist(),
                                       np.asarray(m.vertices).max(0).tolist()]}
                       for k, m in heads.items()}
    front_seg = cv2.imread(str(args.data_dir / 'maps/seg/front.png'), 0) > 127
    front_camera = load_observation_camera(args.data_dir, 'front')
    front_charts = {k: VisibleSurfaceGrowth(front_camera, front_seg.astype(np.int32),
                                           np.zeros((*front_seg.shape, 2)), m)
                    for k, m in heads.items()}
    # 轮廓叠图只比较同一原图标定；不优化相机、不用生成的 front。
    raw = cv2.imread(str(args.data_dir / 'raw_img.png'))
    if raw is None:
        raise ValueError('原图副本 raw_img.png 缺失')
    overlay = cv2.resize(raw, front_seg.shape[::-1])
    for mask, color in [(front_seg, (0, 255, 0)),
                        (front_charts['template'].head_hit, (0, 0, 255)),
                        (front_charts['reference'].head_hit, (255, 255, 0))]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
    cv2.imwrite(str(args.output_dir / 'front_head_contours.png'), overlay)
    report['views'] = {}
    fixed_gaps = {}
    for view in ['front', 'left', 'right', 'back']:
        seg_path = args.data_dir / 'maps/seg' / f'{view}.png'
        seg = cv2.imread(str(seg_path), 0) > 127
        report['input_sha256'][str(seg_path.resolve())] = sha256_file(seg_path)
        camera = load_observation_camera(args.data_dir, view)
        charts = {k: VisibleSurfaceGrowth(camera, seg.astype(np.int32), np.zeros((*seg.shape, 2)), m)
                  for k, m in heads.items()}
        hair_chart = VisibleSurfaceGrowth(camera, seg.astype(np.int32), np.zeros((*seg.shape, 2)), hair)
        template = charts['template']
        covered = template.covered(strands)
        gap = template.target & ~covered
        fixed_gaps[view] = gap
        ids = np.flatnonzero(gap)
        launch = template.head_points[ids] + .0008 * template.normals[ids]
        conflicts = {k: conflict_points(launch, chart) for k, chart in front_charts.items()}
        c0, margin = conflicts['template']
        c1, _ = conflicts['reference']
        reference_distance = distances(head_scenes['reference'], launch, signed=closed['reference'])
        record = {'evidence': 'original_photo' if view == 'front' else 'flux_prior',
                  'fixed_template_gap_pixels': int(gap.sum()),
                  'fixed_template_target_pixels': int(template.target.sum()),
                  'fixed_launch_front_conflict': {'template': int(c0.sum()), 'reference_occluder': int(c1.sum()),
                     'resolved_by_reference_occluder': int((c0 & ~c1).sum()),
                     'introduced_by_reference_occluder': int((~c0 & c1).sum()),
                     'launch_inside_reference_head': int((reference_distance < 0).sum()) if closed['reference'] else None,
                     'resolved_but_inside_reference_head': int((c0 & ~c1 & (reference_distance < 0)).sum()) if closed['reference'] else None,
                     'outside_front_seg_distance_px_quantiles': quantiles(margin[c0]),
                     'outside_more_than_10px': int((c0 & (margin > 10)).sum())},
                  'seg_pixels': int(seg.sum()), 'mesh_misses_in_seg': int((seg & ~hair_chart.head_hit).sum()),
                  'mesh_hit_depth_vs_head': {}}
        hit_mask = seg & hair_chart.head_hit
        for k, chart in charts.items():
            both = hit_mask & chart.head_hit
            behind = both & (hair_chart.head_z < chart.head_z - 1e-5)
            surface_points = hair_chart.head_points[hit_mask.ravel()]
            measured = distances(head_scenes[k], surface_points, signed=closed[k])
            record['mesh_hit_depth_vs_head'][k] = {
                'joint_hit_pixels': int(both.sum()), 'hair_behind_head_pixels': int(behind.sum()),
                'visible_mesh_samples': len(surface_points),
                'mesh_samples_inside_head': int((measured < -.0004).sum()) if closed[k] else None,
                'distance_kind': 'signed' if closed[k] else 'unsigned_open_head',
                'mesh_head_distance_mm_quantiles': quantiles(measured * 1000)}
            if k == 'template':
                colors = [(seg, (60, 60, 60)), (hit_mask, (0, 180, 0)),
                          (behind, (0, 0, 255)), (seg & ~hair_chart.head_hit, (255, 0, 255))]
                write_categories(args.output_dir / f'{view}_mesh_depth.png', seg.shape, colors)
                # 空洞是否聚集在非流形边附近：只记录相关性，不当作因果。
                if bad_tree is not None:
                    gap_hit = gap & hair_chart.head_hit
                    record['gap_mesh_hit_distance_to_nonmanifold_edge_midpoint_mm'] = quantiles(
                        bad_tree.query(hair_chart.head_points[gap_hit.ravel()])[0] * 1000)
        conflict_map = np.zeros(seg.shape, bool)
        conflict_map.ravel()[ids[c0]] = True
        write_categories(args.output_dir / f'{view}_fixed_gap_conflicts.png', seg.shape,
                         [(template.target, (40, 40, 40)), (gap, (0, 180, 255)),
                          (conflict_map, (0, 0, 255))])
        report['views'][view] = record
        print(json.dumps({'stage': view, 'result': record}), flush=True)
    # 固定物理弧长采样；避免旧轨迹密集采样及末端填充影响统计。
    lengths, max_turns, samples, owners = [], [], [], []
    for index, strand in enumerate(strands):
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
        lengths.append(arc[-1])
        grid = np.arange(0., arc[-1], .002)
        points = np.column_stack([np.interp(grid, arc, strand[:, k]) for k in range(3)])
        steps = np.diff(points, axis=0)
        if len(steps) > 1:
            steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-12)
            max_turns.append(float(np.degrees(np.arccos(np.clip((steps[:-1] * steps[1:]).sum(1), -1, 1))).max()))
        else:
            max_turns.append(np.nan)
        body = points[grid >= .006][::2]
        samples.extend(body)
        owners.extend([index] * len(body))
    lengths, max_turns = np.asarray(lengths), np.asarray(max_turns)
    samples, owners = np.asarray(samples), np.asarray(owners)
    separation = distances(head_scenes['template'], samples)
    report['strand_groups'] = {}
    for name, selected in [('preserved_baseline', np.arange(len(strands)) < args.baseline_count),
                           ('added_after_baseline', np.arange(len(strands)) >= args.baseline_count)]:
        body = selected[owners]
        turns = max_turns[selected & np.isfinite(max_turns)]
        report['strand_groups'][name] = {'strands': int(selected.sum()),
             'length_mm_quantiles': quantiles(lengths[selected] * 1000),
             'at_120mm_within_0_01mm': int((selected & (np.abs(lengths - .12) < .00001)).sum()),
             'under_15mm_with_0_01mm_tolerance': int((selected & (lengths < .01499)).sum()),
             'maximum_turn_deg_at_2mm_quantiles': quantiles(turns),
             'strands_with_2mm_turn_over_60deg': int((turns > 60).sum()),
             'body_distance_to_template_mm_quantiles': quantiles(separation[body] * 1000),
             'body_samples_within_4mm_fraction': float((separation[body] < .004).mean()),
             'body_sampling': '4mm arc grid, omit initial 6mm, sample-weighted'}
    print(json.dumps({'stage': 'strands', 'result': report['strand_groups']}), flush=True)
    with np.load(args.run_dir / 'view_evidence.npz') as data:
        labels, camera = data['labels'], data['camera']
    growth = json.loads((args.run_dir / 'report.json').read_text())
    direction_seg = cv2.imread(str(args.data_dir / 'maps/seg' / f"{growth['view']}.png"), 0) > 127
    chart = VisibleSurfaceGrowth(camera, labels, np.zeros((*labels.shape, 2)), head, hair_domain=direction_seg)
    local = LocalGuideField(strands[:growth['existing_root_count']], chart, support_m=growth['guide_support_m'])
    report['direction'] = {'partition_report': growth['partition_report'], 'regions': {},
                           'stop_reasons': dict(Counter(r['stop_reason'] for r in growth['records'])),
                           'rejections': dict(Counter(str(r['rejection']) for r in growth['records']))}
    with np.load(args.run_dir / 'surface_direction_field.npz') as field:
        vertices, normals = field['vertices'], field['normals']
        for label, (tree, directions) in local.fields.items():
            distance, ids = tree.query(vertices, k=min(8, tree.n))
            distance, ids = distance.reshape(len(vertices), -1), ids.reshape(len(vertices), -1)
            weight = 1 / np.maximum(distance, .001) ** 2
            weight /= weight.sum(1, keepdims=True)
            vectors = directions[ids]
            mean = np.sum(weight[..., None] * vectors, axis=1)
            coherence = np.linalg.norm(mean, axis=1)
            moment = np.einsum('nk,nki,nkj->nij', weight, vectors, vectors)
            axial_coherence = np.linalg.eigvalsh(moment)[:, -1]
            active = distance[:, 0] < growth['guide_support_m']
            magnitude = np.linalg.norm(field[f'region_{label}'], axis=1)
            crown = active & (vertices[:, 1] > vertices[:, 1].max() - .2 * np.ptp(vertices[:, 1])) & (normals[:, 1] > .5)
            cancelled = (coherence < .3) & (axial_coherence > .85)
            report['direction']['regions'][str(label)] = {
                'active_vertices': int(active.sum()), 'crown_vertices': int(crown.sum()),
                'directed_neighbor_coherence_quantiles': quantiles(coherence[active]),
                'axial_dominant_eigenvalue_quantiles': quantiles(axial_coherence[active]),
                'opposed_aligned_neighbors_vertices': int((active & cancelled).sum()),
                'opposed_aligned_neighbors_crown_vertices': int((crown & cancelled).sum()),
                'solved_magnitude_under_query_threshold_vertices': int((active & (magnitude < .05)).sum()),
                'solved_magnitude_under_query_threshold_crown_vertices': int((crown & (magnitude < .05)).sum()),
                'cancellation_definition': 'directed mean norm <0.3 AND axis second-moment eigenvalue >0.85; not true root-tip labels'}
    np.savez_compressed(args.output_dir / 'strand_diagnostics.npz', length_m=lengths,
                        max_turn_deg_at_2mm=max_turns, baseline_count=args.baseline_count)
    report['limits'] = [
        '头模替换只改变固定空间点的遮挡判断，不能视为重建或配准修复。',
        '网格第一交点以现有二维 seg 筛选；未证明这些三角形本身属于头发。',
        '有符号头模距离采用 11 射线采样；非流形头发 mesh 不参与有符号距离判断。',
        '方向抵消仅用已有 guide 和保存的场诊断，不提供真实发缝或根尖真值。',
        '所有覆盖量是固定相机像素中心代理，不能替代实体发丝渲染质量。']
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({'stage': 'complete', 'report': str(args.output_dir / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
