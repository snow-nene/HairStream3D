"""统一样例头皮、轴向场、局部长度及原图语义的批量发丝重积分。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.axial_surface_field import AxialSurfaceField
from lib.guide_reconnection import reconnect_guide
from lib.guide_length_evidence import local_length_budgets
from lib.multiview_hair_evidence import classify_evidence
from lib.template_identity import load_template_identity, require_bound_template, sha256_file
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, LocalGuideField, load_observation_camera, audit_strands


class SubjectGuards:
    def __init__(self, charts, side_veto=True, front_soft_boundary_px=0.):
        self.charts = charts
        self.side_veto = side_veto
        self.scene = charts['front'].scene
        self.front_soft_boundary_px = float(front_soft_boundary_px)
        self.front_boundary_distance = {name: distance_transform_edt(~chart.hair_domain)
                                        for name, chart in charts.items()}

    def evidence(self, points):
        """返回统一多视角证据状态，不改变旧的 points/segments 判定。"""
        points = np.asarray(points, float).reshape(-1, 3)
        front = self.charts['front']
        ids, depth, valid = front.pixels(points)
        front_visible = np.zeros(len(points), bool)
        front_hair = np.zeros(len(points), bool)
        boundary = np.full(len(points), np.inf)
        ix = np.flatnonzero(valid)
        if len(ix):
            x, y = ids[ix].T
            front_visible[ix] = ~front.head_hit[y, x] | (depth[ix] >= front.head_z[y, x]-1e-5)
            front_hair[ix] = front.hair_domain[y, x]
            boundary[ix] = self.front_boundary_distance['front'][y, x]
        side_support = np.zeros(len(points), bool)
        for name, chart in self.charts.items():
            if name == 'front':
                continue
            si, sd, sv = chart.pixels(points)
            j = np.flatnonzero(sv)
            if len(j):
                x, y = si[j].T
                visible = ~chart.head_hit[y, x] | (sd[j] >= chart.head_z[y, x]-1e-5)
                side_support[j] |= visible & chart.hair_domain[y, x]
        nearest = front.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        clearance = ((points-nearest['points'].numpy())*nearest['primitive_normals'].numpy()).sum(1)
        return classify_evidence(front_visible=front_visible, front_hair=front_hair,
                                 side_support=side_support, head_clearance=clearance,
                                 boundary_distance_px=boundary)
        self.scene = charts['front'].scene

    def points(self, points, collision=True):
        reason = np.zeros(len(points), np.int32)
        if collision:
            near = self.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
            gap = ((points-near['points'].numpy())*near['primitive_normals'].numpy()).sum(1)
            reason[gap < .00045] = 3
        front_hair_visible = np.zeros(len(points), bool)
        side_veto = np.zeros(len(points), int)
        for view, chart in self.charts.items():
            ids, depth, valid = chart.pixels(points)
            x, y = ids[valid].T
            ix = np.flatnonzero(valid)
            visible = ~chart.head_hit[y, x] | (depth[valid] >= chart.head_z[y, x]-1e-5)
            if view == 'front':
                # 原图语义否决必须沿点自身的连续像素射线查询，不能用
                # 四舍五入后像素中心的头模深度替代（轮廓处会误截断）。
                if not np.allclose(chart.camera[3], [0, 0, 0, 1]):
                    raise ValueError('front 精确遮挡守卫要求正交相机')
                toward_camera = chart.inverse[:3, 2].copy()
                toward_camera /= np.linalg.norm(toward_camera)
                query = points[valid]
                rays = np.c_[query+2*toward_camera,
                             np.broadcast_to(-toward_camera, query.shape)]
                if len(query):
                    hit = self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
                    visible = hit >= 2-.00001
            positive = chart.hair_domain[y, x]
            if view == 'front':
                outside_hair = visible & ~positive
                if self.front_soft_boundary_px > 0:
                    near_boundary = self.front_boundary_distance[view][y, x] <= self.front_soft_boundary_px
                    outside_hair &= ~near_boundary
                reason[ix[outside_hair]] = 4
                front_hair_visible[ix[visible & positive]] = True
                reason[~valid] = 5
            else:
                side_veto[ix[visible & ~positive]] += 1
        # front 已确认头发的区域不受补全先验否决；隐藏区域要求两侧先验共同否决才停止。
        if self.side_veto:
            reason[(side_veto >= 2) & ~front_hair_visible & (reason == 0)] = 6
        return reason

    def segments(self, start, end):
        # 最大积分步为1mm，9点覆盖整段，采样间距不超过0.125mm。
        t = np.linspace(0, 1, 9)
        samples = start[:, None]+(end-start)[:, None]*t[None, :, None]
        codes = self.points(samples.reshape(-1, 3)).reshape(len(start), -1)
        return codes.max(1)


def integrate(field, guards, roots, references, budgets, targets, step=.001, diagnostics=None):
    n = len(roots)
    count = int(np.ceil(budgets.max()/step))+1
    paths = np.empty((n, count, 3), np.float32)
    paths[:, 0] = roots
    sizes = np.ones(n, int)
    lengths = np.zeros(n)
    reasons = guards.points(roots)
    if diagnostics is not None:
        diagnostics['failure_points'] = np.full_like(roots, np.nan)
        diagnostics['failure_points'][reasons != 0] = roots[reasons != 0]

    def record_failure(indices, start, end, failure):
        if diagnostics is None or not np.any(failure):
            return
        bad = np.flatnonzero(failure != 0)
        t = np.linspace(0, 1, 9)
        samples = start[bad, None] + (end-start)[bad, None]*t[None, :, None]
        codes = guards.points(samples.reshape(-1, 3)).reshape(len(bad), 9)
        found = (codes != 0).any(1)
        first = (codes != 0).argmax(1)
        diagnostics['failure_points'][indices[bad[found]]] = samples[np.flatnonzero(found), first[found]]
    position = roots.copy()
    reference = references.copy()
    for iteration in range(count-1):
        active = np.flatnonzero((reasons == 0) & (lengths < budgets-1e-10))
        if not len(active):
            break
        direction, failure = field.query_batch(position[active], reference[active], targets[active])
        reasons[active[failure != 0]] = failure[failure != 0]
        active, direction = active[failure == 0], direction[failure == 0]
        if not len(active):
            continue
        dt = np.minimum(step, budgets[active]-lengths[active])
        midpoint = position[active]+.5*dt[:, None]*direction
        middle_failure = guards.segments(position[active], midpoint)
        record_failure(active, position[active], midpoint, middle_failure)
        tangent, query_failure = field.query_batch(midpoint, reference[active], targets[active])
        failure = np.maximum(middle_failure, query_failure)
        reasons[active[failure != 0]] = failure[failure != 0]
        active, dt, tangent = active[failure == 0], dt[failure == 0], tangent[failure == 0]
        if not len(active):
            continue
        endpoint = position[active]+dt[:, None]*tangent
        failure = guards.segments(position[active], endpoint)
        record_failure(active, position[active], endpoint, failure)
        reasons[active[failure != 0]] = failure[failure != 0]
        active, dt, tangent, endpoint = active[failure == 0], dt[failure == 0], tangent[failure == 0], endpoint[failure == 0]
        paths[active, sizes[active]] = endpoint
        sizes[active] += 1
        lengths[active] += dt
        position[active], reference[active] = endpoint, tangent
        if iteration % 40 == 0:
            print(json.dumps({'step': iteration, 'active': len(active)}), flush=True)
    return [p[:s].copy() for p, s in zip(paths, sizes)], lengths, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--geometry-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--baseline-count', type=int, default=9289)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--preserve-guide-body', action='store_true')
    p.add_argument('--front-soft-boundary-px', type=float, default=0.,
                   help='诊断实验：front hair seg 外侧的不确定带宽度；内部脸部仍为硬约束')
    p.add_argument('--views', nargs='+', choices=['front', 'left', 'right', 'back'],
                   default=['front', 'left', 'right', 'back'])
    a = p.parse_args()
    if 'front' not in a.views:
        raise ValueError('必须包含原图 front')
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按图像聚合')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(a.geometry_dir/'template_head/template_head.npz')
    with np.load(a.geometry_dir/'subject_head_candidate.npz') as d:
        old_v, vertices, faces = d['source_vertices'], d['vertices'], d['faces']
        desired = d['desired_gap']
    head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    old_head = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(old_v), o3d.utility.Vector3iVector(faces))
    old_scene = o3d.t.geometry.RaycastingScene()
    old_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(old_head))
    old_tree = cKDTree(old_v)
    charts = {}
    for view in a.views:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        charts[view] = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view),
                                           seg.astype(np.int32), np.zeros((*seg.shape, 2)), head)
    guards = SubjectGuards(charts, front_soft_boundary_px=a.front_soft_boundary_px)
    with np.load(a.input) as d:
        require_bound_template(d, Path('assets/render_template.blend'))
        dense = d['strands']
        all_roots = dense[:, 0].copy()
        guides, guide_ids, guide_censored = [], [], []
        for source_id, strand in enumerate(dense[:a.baseline_count]):
            keep = np.r_[True, np.linalg.norm(np.diff(strand, axis=0), axis=1)>0]
            strand = strand[keep]
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
            grid = np.arange(0, arc[-1], .002)
            if len(grid) < 3:
                continue
            samples = np.column_stack([np.interp(grid, arc, strand[:, k]) for k in range(3)])
            distance, ids = old_tree.query(samples, k=4)
            weights = 1/np.maximum(distance, .001)**2
            if a.preserve_guide_body:
                near = old_scene.compute_closest_points(o3d.core.Tensor(strand[:1].astype(np.float32)))
                tri = faces[near['primitive_ids'].numpy()[0]]
                uv = near['primitive_uvs'].numpy()[0]
                base = np.array([1-uv.sum(), *uv])@vertices[tri]
                normal = np.cross(vertices[tri[1]]-vertices[tri[0]], vertices[tri[2]]-vertices[tri[0]])
                root = base+.0008*normal/np.linalg.norm(normal)
                samples = reconnect_guide(strand, root)
            else:
                samples += ((vertices-old_v)[ids]*weights[..., None]).sum(1)/weights.sum(1)[:, None]
            codes = guards.points(samples, collision=False)
            stop = np.flatnonzero(codes)
            if len(stop):
                samples = samples[:stop[0]]
            if len(samples) >= 3:
                guides.append(samples)
                guide_ids.append(source_id)
                guide_censored.append(bool(len(stop)))
        del dense
    if not guides:
        raise RuntimeError('无有效局部 guide')
    root_count = len(all_roots) if not a.limit else min(a.limit, len(all_roots))
    ids = np.linspace(0, len(all_roots)-1, root_count).astype(int)
    old_roots = all_roots[ids]
    closest = old_scene.compute_closest_points(o3d.core.Tensor(old_roots.astype(np.float32)))
    triangle = faces[closest['primitive_ids'].numpy()]
    uv = closest['primitive_uvs'].numpy()
    weights = np.c_[1-uv.sum(1), uv]
    root_base = (vertices[triangle]*weights[..., None]).sum(1)
    normals = np.cross(vertices[triangle[:, 1]]-vertices[triangle[:, 0]], vertices[triangle[:, 2]]-vertices[triangle[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    roots = root_base+.0008*normals
    # 方向约束只来自已做语义筛选的 guide；标签1是候选域，不能解释为真实分缝。
    chart = charts['front']
    chart.labels = np.ones_like(chart.labels)
    local = LocalGuideField(guides, chart, support_m=.05)
    offset_query = None
    if a.preserve_guide_body:
        guide_samples = np.concatenate(guides)
        near = chart.scene.compute_closest_points(o3d.core.Tensor(guide_samples.astype(np.float32)))
        guide_gap = ((guide_samples-near['points'].numpy())*near['primitive_normals'].numpy()).sum(1)
        gap_tree = cKDTree(guide_samples)
        def offset_query(points):
            distance, near_ids = gap_tree.query(points, k=8)
            weight = 1/np.maximum(distance, .002)**2
            return np.clip((guide_gap[near_ids]*weight).sum(1)/weight.sum(1), .0012, .04)
    field = AxialSurfaceField(head, local, chart, offset_query=offset_query)
    np.savez_compressed(a.output_dir/'axial_field.npz', vertices=vertices,
                        **{f'region_{k}': v for k, v in field.fields.items()})
    guide_roots = np.array([s[0] for s in guides])
    guide_lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in guides])
    usable = (guide_lengths >= .004) & (guide_lengths <= .18)
    tree = cKDTree(guide_roots[usable])
    distance, nearest = tree.query(roots, k=min(8, int(usable.sum())))
    weights = 1/np.maximum(distance, .002)**2
    budgets, length_supported = local_length_budgets(roots, guide_roots, guide_lengths, guide_censored)
    np.savez_compressed(a.output_dir/'guide_length_evidence.npz', source_root_ids=guide_ids,
                        roots=guide_roots, length_m=guide_lengths, censored=guide_censored)
    directions = np.array([(s[2]-s[0])/np.linalg.norm(s[2]-s[0]) for s in guides])[usable]
    reference = (directions[nearest]*weights[..., None]).sum(1)/weights.sum(1)[:, None]
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    local_gap = desired[triangle].mean(1)
    # 分层位置是确定性几何先验，避免全部发丝贴在统一3mm曲面。
    layer = .35+.55*((ids*.6180339887498949) % 1)
    targets = np.maximum(.0012, local_gap*layer)
    print(json.dumps({'guides': len(guides), 'roots': len(roots), 'field': field.report,
                       'budget_mm_quantiles': (np.quantile(budgets, [0, .5, 1])*1000).tolist()}), flush=True)
    # 两条短分支都接受相同的整段守卫，正式重积分时从根点重新开始。
    _, forward, _ = integrate(field, guards, roots, reference, np.full(len(roots), .02), targets)
    _, reverse, _ = integrate(field, guards, roots, -reference, np.full(len(roots), .02), targets)
    reverse_selected = reverse > forward+.002
    reference[reverse_selected] *= -1
    paths, lengths, stops = integrate(field, guards, roots, reference, budgets, targets)
    stops[~length_supported] = 8
    reconnected = np.zeros(len(paths), bool)
    if a.preserve_guide_body:
        lookup = {source_id: strand for source_id, strand in zip(guide_ids, guides)}
        for i, source_id in enumerate(ids):
            if source_id not in lookup:
                continue
            strand = lookup[source_id]
            # 重采样保证非均匀根部连接也满足整段守卫的125微米采样间距。
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
            grid = np.linspace(0, arc[-1], int(np.ceil(arc[-1]/.001))+1)
            strand = np.column_stack([np.interp(grid, arc, strand[:, k]) for k in range(3)])
            codes = guards.segments(strand[:-1], strand[1:])
            failure = np.flatnonzero(codes)
            stop = codes[failure[0]] if len(failure) else 7
            if len(failure):
                strand = strand[:failure[0]+1]
            length = np.linalg.norm(np.diff(strand, axis=0), axis=1).sum()
            if length >= .004:
                paths[i], lengths[i], stops[i], reconnected[i] = strand, length, stop, True
    selected = lengths >= .004-1e-7
    pieces = [path for path, keep in zip(paths, selected) if keep]
    if not pieces:
        raise RuntimeError('所有根均被拒绝')
    collision = audit_strands(pieces, head)
    report = {'template_identity': identity, 'source': str(a.input.resolve()), 'source_sha256': sha256_file(a.input),
              'root_candidates': len(roots), 'accepted': len(pieces), 'rejected_short': int((~selected).sum()),
              'stops': dict(Counter(map(int, stops))),
              'stop_names': {'0': 'local_guide_length', '1': 'axial_support', '2': 'axial_ambiguity',
                             '3': 'head_clearance', '4': 'front_semantic_exit', '5': 'outside_front_image',
                             '6': 'two_prior_exit', '7': 'source_guide_end',
                             '8': 'no_local_uncensored_length'},
              'reverse_branches': int(reverse_selected.sum()), 'collision': collision,
              'budget_mm_quantiles': (np.quantile(budgets, [0, .1, .5, .9, 1])*1000).tolist(),
              'accepted_length_mm_quantiles': (np.quantile(lengths[selected], [0, .1, .5, .9, 1])*1000).tolist(),
              'at_120mm': int(np.sum(np.abs(lengths[selected]-.12)<1e-5)), 'field': field.report,
              'reconnected_guide_bodies': int(reconnected.sum()),
              'roots_without_local_uncensored_length': int((~length_supported).sum()),
              'body_policy': 'preserve_smoothed_guide_body' if a.preserve_guide_body else 'reintegrate_all',
              'status': 'requires_geometry_and_render_validation'}
    (a.output_dir/'report.json').write_text(json.dumps(report, indent=2))
    if not collision['passed']:
        raise RuntimeError('全量碰撞审计失败，未导出发丝')
    width = max(map(len, pieces))
    packed = np.empty((len(pieces), width, 3), np.float32)
    for i, strand in enumerate(pieces):
        packed[i, :len(strand)] = strand
        packed[i, len(strand):] = strand[-1]
    np.savez_compressed(a.output_dir/'all_root_prefixes.npz', strands=packed,
                        valid_point_counts=np.array(list(map(len, pieces))), source_root_ids=ids[selected],
                        template_blend_sha256=np.array(identity['template_sha256']))
    np.savez_compressed(a.output_dir/'root_diagnostics.npz', source_root_ids=ids, roots=roots,
                        accepted=selected, budgets=budgets, length_m=lengths, stop_code=stops)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
