"""保留已验收形状，裁去原图外可见发梢，并在合法缺口重积分新根。"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.axial_surface_field import AxialSurfaceField
from lib.coverage_acceptance import accept_coverage_candidates
from lib.guide_length_evidence import local_length_budgets
from lib.template_identity import load_template_identity, require_bound_template, sha256_file
from scripts.recon_3d.regrow_subject_strands import SubjectGuards, integrate
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera, sample_polyline, audit_strands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--geometry-dir', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--extra-roots', type=int, default=4000)
    p.add_argument('--baseline-count', type=int, default=9289)
    p.add_argument('--front-boundary-guidance', action='store_true',
                   help='实验：用原图边界及遮挡余量对方向作有限角度引导')
    p.add_argument('--max-guide-offset-m', type=float, default=.025,
                   help='隔离实验：邻近发丝离头距离上限，默认25mm')
    p.add_argument('--minimum-added-length-m', type=float, default=.004,
                   help='新增发丝覆盖验收最短长度；短发隔离实验使用1mm')
    p.add_argument('--accept-crown-coverage', action='store_true')
    p.add_argument('--disable-side-veto', action='store_true', help='隔离实验：仅对新增根及生长关闭侧视共同否决')
    p.add_argument('--front-soft-boundary-px', type=float, default=0.,
                   help='诊断实验：front hair seg 外侧不确定带宽度')
    p.add_argument('--views', nargs='+', choices=['front', 'left', 'right', 'back'],
                   default=['front', 'left', 'right', 'back'])
    a = p.parse_args()
    if 'front' not in a.views or a.extra_roots <= 0:
        raise ValueError('必须包含原图 front，补根预算必须为正数')
    if not .0012 <= a.max_guide_offset_m <= .025:
        raise ValueError('离头距离上限应在1.2至25mm之间')
    if not .001 <= a.minimum_added_length_m <= .004:
        raise ValueError('新增验收长度须在1至4mm之间')
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出必须按图像聚合')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(a.geometry_dir/'template_head/template_head.npz')
    with np.load(a.geometry_dir/'template_head/template_head.npz') as d:
        vertices, faces = d['vertices'], d['faces']
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    charts = {}
    for view in a.views:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        charts[view] = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view), seg.astype(int),
                                          np.zeros((*seg.shape, 2)), mesh)
    guards = SubjectGuards(charts, front_soft_boundary_px=a.front_soft_boundary_px)
    with np.load(a.input_dir/'all_root_prefixes.npz') as d:
        require_bound_template(d, Path(identity['template_path']))
        strands = [s[np.r_[True, np.linalg.norm(np.diff(s, axis=0), axis=1)>0]] for s in d['strands']]
        source_ids = d['source_root_ids'].copy()
    cleaned, cleaned_ids, trimmed = [], [], 0
    for strand, source_id in zip(strands, source_ids):
        samples = sample_polyline(strand, .000125)
        failure = np.flatnonzero(guards.points(samples))
        if len(failure):
            samples = samples[:failure[0]]
            trimmed += 1
            strand = samples[np.r_[np.arange(0, max(len(samples)-1, 0), 8), len(samples)-1]] if len(samples)>1 else samples
            if len(strand) > 1:
                bad_segment = np.flatnonzero(guards.segments(strand[:-1], strand[1:]))
                if len(bad_segment):
                    strand = strand[:bad_segment[0]+1]
        if len(strand) > 1 and np.linalg.norm(np.diff(strand, axis=0), axis=1).sum() >= .004:
            cleaned.append(strand)
            cleaned_ids.append(source_id)
    del strands
    # 固定输入清理阶段；侧视消融只改变候选根与新增生长。
    if a.disable_side_veto:
        guards = SubjectGuards(charts, side_veto=False,
                               front_soft_boundary_px=a.front_soft_boundary_px)
    print(json.dumps({'cleaned': len(cleaned), 'trimmed': trimmed}), flush=True)
    candidates = vertices+.0008*np.asarray(mesh.vertex_normals)
    admissible = guards.points(candidates) == 0
    score = np.zeros(len(candidates))
    for view, chart in charts.items():
        covered = chart.covered(cleaned)
        pixel, depth, valid = chart.pixels(candidates)
        ids = np.flatnonzero(valid)
        x, y = pixel[valid].T
        gap = chart.target[y, x] & ~covered[y, x]
        visible = depth[valid] >= chart.head_z[y, x]-1e-4
        score[ids[gap & visible]] += 3 if view == 'front' else 1
    # 顶部只作为几何冠部覆盖先验；不将它当作新增观测图片。
    tree = cKDTree(np.concatenate([sample_polyline(s, .002) for s in cleaned]))
    distance = tree.query(candidates)[0]
    score += ((vertices[:, 1] > vertices[:, 1].max()-.05) & (distance > .0025))*.5
    eligible = np.flatnonzero(admissible & (score > 0))
    order = eligible[np.argsort(-(score[eligible]+np.minimum(distance[eligible], .015)*10), kind='stable')]
    candidate_tree = cKDTree(candidates)
    blocked = np.zeros(len(candidates), bool)
    selected = []
    for index in order:
        if blocked[index]:
            continue
        selected.append(index)
        blocked[candidate_tree.query_ball_point(candidates[index], .0009)] = True
        if len(selected) >= a.extra_roots:
            break
    selected = np.array(selected, int)
    if not len(selected):
        raise RuntimeError('无合法补生长根点')
    roots = candidates[selected]
    guides = [s for s, source_id in zip(cleaned, cleaned_ids) if 0 <= source_id < a.baseline_count and len(s)>2]
    guide_roots = np.array([s[0] for s in guides])
    lengths = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in guides])
    usable = (lengths >= .004) & (lengths <= .18)
    guide_tree = cKDTree(guide_roots[usable])
    d, nearest = guide_tree.query(roots, k=min(8, int(usable.sum())))
    weights = 1/np.maximum(d, .002)**2
    evidence_path = a.input_dir/'guide_length_evidence.npz'
    if not evidence_path.exists():
        raise ValueError('缺少长度截断来源记录，请先重新运行 regrow 阶段')
    with np.load(evidence_path) as evidence:
        budgets, length_supported = local_length_budgets(roots, evidence['roots'],
                                                        evidence['length_m'], evidence['censored'])
    directions = np.array([(s[min(3,len(s)-1)]-s[0])/np.linalg.norm(s[min(3,len(s)-1)]-s[0]) for s in guides])[usable]
    references = (directions[nearest]*weights[..., None]).sum(1)
    references /= np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    field = AxialSurfaceField.__new__(AxialSurfaceField)
    field.tree, field.chart = cKDTree(vertices), charts['front']
    with np.load(a.input_dir/'axial_field.npz') as d:
        if d['vertices'].shape != vertices.shape or not np.allclose(d['vertices'], vertices, rtol=0, atol=2e-7):
            raise ValueError('轴向场与绑定头模顶点不一致')
        field.fields = {int(k[7:]): d[k] for k in d.files if k.startswith('region_')}
    samples = np.concatenate([sample_polyline(s, .002) for s in guides])
    near = field.chart.scene.compute_closest_points(o3d.core.Tensor(samples.astype(np.float32)))
    gap = ((samples-near['points'].numpy())*near['primitive_normals'].numpy()).sum(1)
    sample_tree = cKDTree(samples)
    def offset_query(points):
        d, ids = sample_tree.query(points, k=8)
        w = 1/np.maximum(d, .002)**2
        return np.clip((gap[ids]*w).sum(1)/w.sum(1), .0012, a.max_guide_offset_m)
    field.offset_query = offset_query
    if a.front_boundary_guidance:
        from lib.front_boundary_guidance import FrontBoundaryGuidance
        field.boundary_guidance = FrontBoundaryGuidance(charts['front'])
    targets = np.full(len(roots), .002)
    print(json.dumps({'eligible': len(eligible), 'selected_roots': len(roots)}), flush=True)
    _, forward, _ = integrate(field, guards, roots, references, np.minimum(budgets, .02), targets)
    _, reverse, _ = integrate(field, guards, roots, -references, np.minimum(budgets, .02), targets)
    references[reverse > forward+.002] *= -1
    growth_diagnostics = {}
    paths, added_lengths, stops = integrate(field, guards, roots, references, budgets, targets,
                                           diagnostics=growth_diagnostics)
    raw_stops = stops.copy()
    stops[~length_supported] = 8
    acceptance_charts = dict(charts)
    if a.accept_crown_coverage:
        center = (vertices.min(0)+vertices.max(0))/2
        scale = 1.8/max(np.ptp(vertices[:,0]), np.ptp(vertices[:,2]))
        camera = np.array([[scale,0,0,-scale*center[0]], [0,0,scale,-scale*center[2]],
                           [0,1,0,-center[1]], [0,0,0,1]])
        top = VisibleSurfaceGrowth(camera, np.ones((512,512),int), np.zeros((512,512,2)), mesh)
        top.target &= (top.head_points[:,1].reshape(512,512)>vertices[:,1].max()-.2*np.ptp(vertices[:,1]))
        top.target &= top.normals[:,1].reshape(512,512)>.5
        acceptance_charts['top'] = top
    accepted, coverage_gains = accept_coverage_candidates(paths, added_lengths, acceptance_charts, cleaned,
                                                         minimum_length=a.minimum_added_length_m)
    added = [s for s, keep in zip(paths, accepted) if keep]
    pieces = cleaned+added
    collision = audit_strands(pieces, mesh)
    report = {'source_sha256': sha256_file(a.input_dir/'all_root_prefixes.npz'), 'template_identity': identity,
              'input_strands': len(source_ids), 'trimmed_by_strict_photo_guard': trimmed,
              'retained': len(cleaned), 'new_candidates': len(roots), 'new_accepted': len(added),
              'exported': len(pieces), 'collision': collision,
              'ablation': {'accept_crown_coverage':a.accept_crown_coverage,
                           'front_boundary_guidance': a.front_boundary_guidance,
                           'max_guide_offset_m': a.max_guide_offset_m,
                           'minimum_added_length_m': a.minimum_added_length_m,
                           'disable_side_veto_for_new_growth':a.disable_side_veto},
              'rejected_no_coverage_gain': int(((added_lengths >= a.minimum_added_length_m) & ~accepted).sum()),
              'new_target_pixels_sum_across_views': int(coverage_gains[accepted].sum()),
              'roots_without_local_uncensored_length': int((~length_supported).sum()),
              'semantics': 'front_original_photo_strict_including_outside_head_silhouette;side_images_priors_only'}
    (a.output_dir/'report.json').write_text(json.dumps(report, indent=2))
    if not collision['passed']:
        raise RuntimeError('碰撞审计未通过')
    width = max(map(len, pieces))
    packed = np.empty((len(pieces), width, 3), np.float32)
    for i, s in enumerate(pieces):
        packed[i, :len(s)] = s
        packed[i, len(s):] = s[-1]
    np.savez_compressed(a.output_dir/'all_root_prefixes.npz', strands=packed,
        source_root_ids=np.r_[cleaned_ids, -(selected[accepted]+1)],
        template_blend_sha256=np.array(identity['template_sha256']))
    np.savez_compressed(a.output_dir/'root_diagnostics.npz', vertex_ids=selected,
                        accepted=accepted, length_m=added_lengths, stop_code=stops,
                        integration_stop_code=raw_stops, **growth_diagnostics,
                        coverage_gain_pixels=coverage_gains)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
