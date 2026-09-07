#!/usr/bin/env python3
"""从标定视角的可见头皮缺口重新积分；二维候选分区不等同可信三维发缝。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.coverage_budget import allocate_coverage_roots
from lib.supplemental_growth import execute_supplemental_growth
from lib.template_identity import load_template_identity, require_bound_template
from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration
from scripts.vis.audit_strand_depth_partitions import (
    compute_partition_evidence, clean_candidate_edge, partition_from_barrier,
)


def sample_polyline(points, spacing):
    points = np.asarray(points, float)
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    positions = np.unique(np.r_[length, np.arange(0, length[-1], spacing)])
    return np.stack([np.interp(positions, length, points[:, i]) for i in range(3)], axis=1)


def audit_strands(strands, mesh, spacing=.000125, clearance=.0004):
    """独立射线占据及无符号距离审计，覆盖已有和新增的全部线段。"""
    if not mesh.is_watertight():
        raise ValueError('watertight mesh required')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    inside, total, minimum = 0, 0, float('inf')
    for index, strand in enumerate(strands):
        if not len(strand) or not np.isfinite(strand).all():
            raise ValueError(f'invalid strand {index}')
        samples = sample_polyline(strand, spacing)
        tensor = o3d.core.Tensor(samples.astype(np.float32))
        inside += int(scene.compute_occupancy(tensor, nsamples=11).numpy().sum())
        minimum = min(minimum, float(scene.compute_distance(tensor).numpy().min()))
        total += len(samples)
    return {'passed': bool(inside == 0 and minimum >= clearance and total > 0),
            'inside_points': inside, 'minimum_distance_m': minimum if total else None,
            'sampled_points': total, 'sample_spacing_m': spacing, 'clearance_m': clearance,
            'strand_count': len(strands), 'occupancy_rays': 11}


def pack_supplemental_strands(original, additions):
    """原始数组逐点保留，仅在新增发丝更长时补齐原始终点。"""
    width = max([original.shape[1]] + [len(p) for p in additions])
    packed = np.empty((len(original) + len(additions), width, 3), dtype=original.dtype)
    packed[:len(original), :original.shape[1]] = original
    packed[:len(original), original.shape[1]:] = original[:, -1, None]
    for index, points in enumerate(additions, start=len(original)):
        packed[index, :len(points)] = points
        packed[index, len(points):] = points[-1]
    return packed


def template_growth_domain(pixels, closing_radius=12):
    """Close small baseline hair holes without extending hair onto exposed face.

    This is a conservative template-render envelope, not anatomical scalp truth.
    Input is BGR from the independent red-head/green-hair render.
    """
    hair = ((pixels[..., 1] > 127) & (pixels[..., 1] > pixels[..., 2])
            & (pixels[..., 1] > pixels[..., 0])).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * closing_radius + 1,) * 2)
    return cv2.morphologyEx(hair, cv2.MORPH_CLOSE, kernel).astype(bool)


def guard_template_domain(start, end, coverage, allowed):
    samples = sample_polyline(np.asarray([start, end]), .000125)
    ids, _, valid = coverage.pixels(samples)
    if not valid.all() or not allowed[ids[:, 1].clip(0, allowed.shape[0]-1),
                                     ids[:, 0].clip(0, allowed.shape[1]-1)].all():
        return 'template_hairline_exit'
    return None


class LocalGuideField:
    """Reconstruct directed local tangents; never translate a donor trajectory.

    Labels are the source chart's candidate labels. Missing local support or
    opposing flow remains unresolved instead of averaging across a parting.
    """
    def __init__(self, strands, chart, support_m=.012):
        self.chart, self.support_m = chart, support_m
        points, directions, labels = [], [], []
        for strand in strands:
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
            keep = np.r_[True, np.diff(distance) > 0]
            distance, strand = distance[keep], strand[keep]
            grid = np.arange(0., distance[-1], .002)
            samples = np.stack([np.interp(grid, distance, strand[:, axis]) for axis in range(3)], axis=1)
            if len(samples) < 3:
                continue
            tangent = np.diff(samples, axis=0)
            magnitude = np.linalg.norm(tangent, axis=1)
            midpoint = .5 * (samples[:-1] + samples[1:])
            ids, _, valid = chart.pixels(midpoint)
            identity = np.zeros(len(ids), int)
            identity[valid] = chart.labels[ids[valid, 1], ids[valid, 0]]
            keep = valid & (identity > 0) & (magnitude > 1e-8)
            points.extend(midpoint[keep])
            directions.extend(tangent[keep] / magnitude[keep, None])
            labels.extend(identity[keep])
        self.fields = {}
        points, directions, labels = np.asarray(points), np.asarray(directions), np.asarray(labels)
        for label in np.unique(labels):
            keep = labels == label
            self.fields[int(label)] = (cKDTree(points[keep]), directions[keep])

    def query(self, point, partition):
        if partition not in self.fields:
            return np.zeros(3), 'no_same_region_guides'
        tree, directions = self.fields[partition]
        distance, ids = tree.query(point, k=min(12, tree.n), distance_upper_bound=self.support_m)
        distance, ids = np.atleast_1d(distance), np.atleast_1d(ids)
        keep = np.isfinite(distance)
        if keep.sum() < 3:
            return np.zeros(3), 'insufficient_local_guides'
        vectors = directions[ids[keep]]
        weights = 1 / np.maximum(distance[keep], .001)**2
        vector = np.sum(weights[:, None] * vectors, axis=0) / weights.sum()
        coherence = np.linalg.norm(vector)
        if coherence < .7:
            return np.zeros(3), 'ambiguous_guide_flow'
        vector /= coherence
        closest = self.chart.scene.compute_closest_points(o3d.core.Tensor(point[None].astype(np.float32)))
        normal = closest['primitive_normals'].numpy()[0]
        separation = float((point - closest['points'].numpy()[0]) @ normal)
        if separation < self.chart.clearance:
            component = float(vector @ normal)
            outward = max(component, np.clip((self.chart.clearance-separation)/.004, 0, .5))
            vector += (outward-component)*normal
        return vector, None


def load_observation_camera(data_dir, view):
    """Front maps belong to the original photo, not the generated GLB render."""
    if view == 'front':
        from scripts.recon_3d.recon3D import load_calib
        path = Path(data_dir) / 'maps' / 'param' / 'front.npy'
        camera = load_calib(str(path), loadSize=1024)
        return camera.numpy() if hasattr(camera, 'numpy') else np.asarray(camera)
    return load_blender_view_calibration(str(data_dir), view)[0]


class VisibleSurfaceGrowth:
    def __init__(self, camera, labels, axial, mesh, clearance=.0008):
        self.camera = np.asarray(camera, float)
        self.inverse = np.linalg.inv(camera)
        self.labels, self.axial = labels, axial
        self.h, self.w = labels.shape
        self.clearance = clearance
        self.scene = o3d.t.geometry.RaycastingScene()
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene.add_triangles(tensor_mesh)
        # A distinct ray-parity query supplies the independent collision audit.
        self.audit_scene = o3d.t.geometry.RaycastingScene()
        self.audit_scene.add_triangles(tensor_mesh)
        self.cache = {}
        yy, xx = np.indices(labels.shape)
        uv = np.stack([2 * xx / (self.w - 1) - 1, 2 * yy / (self.h - 1) - 1], -1).reshape(-1, 2)
        vertices = np.asarray(mesh.vertices)
        near = (np.c_[vertices, np.ones(len(vertices))] @ self.camera.T)[:, 2].max() + 1
        perspective = not np.allclose(self.camera[3], [0, 0, 0, 1])
        near = 1. if perspective else near
        far = -1. if perspective else near - 1.
        origins_h = np.c_[uv, np.full(len(uv), near), np.ones(len(uv))] @ self.inverse.T
        ends_h = np.c_[uv, np.full(len(uv), far), np.ones(len(uv))] @ self.inverse.T
        origins = origins_h[:, :3] / origins_h[:, 3:]
        ends = ends_h[:, :3] / ends_h[:, 3:]
        rays = ends - origins
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        hit = self.scene.cast_rays(o3d.core.Tensor(np.c_[origins, rays].astype(np.float32)))
        distances = hit['t_hit'].numpy()
        self.head_hit = np.isfinite(distances).reshape(labels.shape)
        self.head_points = origins + np.where(np.isfinite(distances), distances, 0)[:, None] * rays
        self.normals = hit['primitive_normals'].numpy()
        self.incidence = -np.sum(self.normals * rays, axis=1).reshape(labels.shape)
        self.head_z = self.project(self.head_points)[1].reshape(labels.shape)
        self.head_z[~self.head_hit] = -np.inf
        self.target = (labels > 0) & self.head_hit & (self.incidence > .3)

    def project(self, points):
        clip = np.c_[points, np.ones(len(points))] @ self.camera.T
        clip = clip / clip[:, 3:]
        uv = (clip[:, :2] + 1) * np.array([self.w - 1, self.h - 1]) / 2
        return uv, clip[:, 2]

    def pixels(self, points):
        uv, depth = self.project(points)
        ids = np.rint(uv).astype(int)
        valid = ((ids >= 0) & (ids < [self.w, self.h])).all(axis=1)
        return ids, depth, valid

    def mask(self, strand):
        key = hashlib.sha256(np.asarray(strand).tobytes()).digest()
        if key in self.cache:
            return self.cache[key]
        samples = sample_polyline(strand, .00025)
        ids, depth, valid = self.pixels(samples)
        ids, depth = ids[valid], depth[valid]
        visible = depth >= self.head_z[ids[:, 1], ids[:, 0]] - 1e-5
        ids = ids[visible]
        mask = np.zeros((self.h, self.w), np.uint8)
        mask[ids[:, 1], ids[:, 0]] = 1
        # Pixel-centre approximation, used for proposal selection, not final render proof.
        flat = np.flatnonzero(mask)
        self.cache[key] = flat
        return flat

    def covered(self, strands):
        flat = np.zeros(self.h * self.w, bool)
        for strand in strands:
            flat[self.mask(strand)] = True
        return flat.reshape(self.h, self.w)

    def score(self, strands):
        return float(np.count_nonzero(self.covered(strands) & self.target) / max(1, self.target.sum()))

    def query(self, point, partition):
        ids, _, valid = self.pixels(point[None])
        if not valid[0]:
            return np.zeros(3), 'outside_image'
        x, y = ids[0]
        if self.labels[y, x] != partition:
            return np.zeros(3), 'view_partition_boundary'
        axial = self.axial[y, x]
        if np.linalg.norm(axial) < .5:
            return np.zeros(3), 'unknown_direction'
        theta = .5 * np.arctan2(axial[1], axial[0])
        direction = np.array([np.cos(theta), np.sin(theta)])
        # Axial evidence has no root-to-tip sign; gravity chooses the initial branch.
        if direction[1] < 0:
            direction *= -1
        closest = self.scene.compute_closest_points(o3d.core.Tensor(point[None].astype(np.float32)))
        normal = closest['primitive_normals'].numpy()[0]
        distance = float((point - closest['points'].numpy()[0]) @ normal)
        base = self.inverse[:3, :2] @ (2 * direction / [self.w - 1, self.h - 1])
        ray = self.inverse[:3, 2]
        denom = float(normal @ ray)
        if abs(denom) < .15 * np.linalg.norm(ray):
            return np.zeros(3), 'grazing_surface'
        tangent = base - ray * (normal @ base) / denom
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        tangent += normal * np.clip((self.clearance - distance) / .002, -.25, .5)
        return tangent, None

    def guard(self, start, end, partition):
        points = sample_polyline(np.array([start, end]), .000125)
        ids, _, valid = self.pixels(points)
        if not valid.all():
            return 'outside_image'
        if np.any(self.labels[ids[:, 1], ids[:, 0]] != partition):
            return 'view_partition_boundary'
        closest = self.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        distance = np.sum((points - closest['points'].numpy()) * closest['primitive_normals'].numpy(), axis=1)
        if np.any(distance < .00045):
            return 'head_clearance'
        return None

    def audit(self, strand):
        points = sample_polyline(strand, .0000625)
        tensor = o3d.core.Tensor(points.astype(np.float32))
        occupied = self.audit_scene.compute_occupancy(tensor, nsamples=11).numpy()
        distance = self.audit_scene.compute_distance(tensor).numpy()
        minimum = float(distance.min())
        return {'passed': bool(not occupied.any() and minimum >= .0004),
                'inside_points': int(occupied.sum()), 'minimum_distance_m': minimum,
                'sample_spacing_m': .0000625, 'sampled_points': len(points)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--view', choices=['front', 'left', 'right', 'back'], default='back')
    parser.add_argument('--budget', type=int, default=300)
    parser.add_argument('--direction-mode', choices=['image_tangent', 'local_guides'], default='image_tangent')
    parser.add_argument('--surface-offset-m', type=float, default=.0008,
                        help='根点仍为 0.8 mm 间隙，方向场逐渐向外达到该表面距离')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--coverage-camera', type=Path,
                        help='实际模板视角相机；与方向证据视角分开')
    parser.add_argument('--visibility-render', type=Path,
                        help='audit_visible_head_coverage 生成的同相机平色图，仅从红色露头模区域补根')
    args = parser.parse_args()
    if not np.isfinite(args.surface_offset_m) or not .0008 <= args.surface_offset_m <= .004:
        raise ValueError('surface offset must be between 0.8 and 4 mm')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_template_identity(args.head)
    with np.load(args.input) as data:
        require_bound_template(data, Path(identity['template_path']))
        old = data['strands']
        existing = [p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 0]] for p in old]
    del old
    with np.load(args.head) as data:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']), o3d.utility.Vector3iVector(data['faces']))
    if not mesh.is_watertight():
        raise ValueError('template head must be watertight')
    if args.audit_only:
        report = audit_strands(existing, mesh)
        report['template_identity'] = identity
        (args.output_dir / 'audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if not report['passed']:
            raise RuntimeError('independent collision audit failed')
        return
    maps = args.data_dir / 'maps'
    seg = cv2.imread(str(maps / 'seg' / f'{args.view}.png'), 0) > 127
    strand_map = cv2.cvtColor(cv2.imread(str(maps / 'strand_map' / f'{args.view}.png')), cv2.COLOR_BGR2RGB)
    depth = np.load(maps / 'depth_map' / f'{args.view}.npy').squeeze()
    evidence = compute_partition_evidence(strand_map, depth, seg)
    edge, threshold = clean_candidate_edge(evidence['fused'], evidence['valid'], .97)
    labels, partition_report = partition_from_barrier(seg & evidence['valid'], edge)
    camera = load_observation_camera(args.data_dir, args.view)
    chart = VisibleSurfaceGrowth(camera, labels, evidence['axial'], mesh, clearance=args.surface_offset_m)
    coverage = chart
    if args.coverage_camera:
        with np.load(args.coverage_camera) as data:
            require_bound_template(data, Path(identity['template_path']))
            shape = tuple(data['image_shape'])
            coverage = VisibleSurfaceGrowth(data['camera'], np.ones(shape, np.int32),
                                             np.zeros((*shape, 2)), mesh)
        ids, _, valid = chart.pixels(coverage.head_points)
        transferred = np.zeros(len(ids), np.int32)
        transferred[valid] = labels[ids[valid, 1], ids[valid, 0]]
        coverage.labels = transferred.reshape(shape)
        coverage.target &= coverage.labels > 0
    covered = coverage.covered(existing)
    roots = np.asarray([s[0] for s in existing])
    candidates = coverage.head_points + .0008 * coverage.normals
    distances = cKDTree(roots).query(candidates)[0].reshape(coverage.target.shape)
    eligible = coverage.target & ~covered & (distances < .015)
    allowed = None
    if args.visibility_render:
        if not args.coverage_camera:
            raise ValueError('visibility render requires its template camera')
        pixels = cv2.imread(str(args.visibility_render), cv2.IMREAD_COLOR)
        if pixels is None or pixels.shape[:2] != coverage.target.shape:
            raise ValueError('visibility render dimensions must match coverage camera')
        exposed = (pixels[..., 2] > 127) & (pixels[..., 2] > pixels[..., 1])
        allowed = template_growth_domain(pixels)
        coverage.target &= allowed
        eligible &= allowed
        eligible &= exposed
        coverage.target &= exposed
        cv2.imwrite(str(args.output_dir / 'template_growth_domain.png'), allowed.astype(np.uint8) * 255)
    proposal = allocate_coverage_roots(candidates, coverage.labels.ravel(), eligible.ravel(), roots,
                                       budget=args.budget, min_distance=.0015)
    print(json.dumps({'target_pixels': int(coverage.target.sum()), 'gap_pixels': int((coverage.target & ~covered).sum()),
                      'candidates': int(eligible.sum()), 'selected': len(proposal['indices'])}), flush=True)
    coverage_state = {'mask': coverage.covered(existing)}
    target_count = max(1, int(coverage.target.sum()))
    def coverage_delta(_current, candidate):
        candidate_mask = np.zeros(coverage.target.size, bool)
        candidate_mask[coverage.mask(candidate)] = True
        return float(np.count_nonzero(coverage.target.ravel() & candidate_mask & ~coverage_state['mask'].ravel()) / target_count)
    def coverage_commit(candidate):
        coverage_state['mask'].ravel()[coverage.mask(candidate)] = True
    def trajectory_quality(current, candidate):
        endpoints = [p[-1] for p in current]
        if not endpoints:
            return True
        return bool(cKDTree(np.asarray(endpoints)).query(candidate[-1])[0] >= .0035)

    def guarded_segment(start, end, partition):
        if allowed is not None:
            reason = guard_template_domain(start, end, coverage, allowed)
            if reason:
                return reason
        return chart.guard(start, end, partition)

    field_query = chart.query
    if args.direction_mode == 'local_guides':
        local_field = LocalGuideField(existing, chart)
        field_query = local_field.query
    result = execute_supplemental_growth({'roots': proposal}, query_field=field_query,
        check_segment=guarded_segment, audit_trajectory=chart.audit, coverage_score=coverage.score,
        existing_strands=existing, step_m=.0005, length_m=.12, min_length_m=.025,
        coverage_delta=coverage_delta, coverage_commit=coverage_commit,
        trajectory_quality=trajectory_quality)
    extra = result.pop('strands')
    after = coverage.covered(existing + extra)
    for name, mask in [('target', coverage.target), ('gap_before', coverage.target & ~covered),
                       ('gap_after', coverage.target & ~after), ('barrier', edge)]:
        cv2.imwrite(str(args.output_dir / f'{name}.png'), mask.astype(np.uint8) * 255)
    result.update({'status': 'candidate_requires_multiview_render_validation', 'view': args.view,
                   'partition_semantics': 'view_local_axial_depth_candidate_barriers_not_trusted_3d_parting',
                   'partition_report': partition_report, 'edge_threshold': threshold,
                   'template_identity': identity, 'input': str(args.input.resolve()),
                   'coverage_camera': str(args.coverage_camera),
                   'visibility_render': str(args.visibility_render),
                   'surface_offset_m': args.surface_offset_m,
                   'direction_mode': args.direction_mode,
                   'target_pixels': int(coverage.target.sum()), 'gap_pixels_before': int((coverage.target & ~covered).sum()),
                   'gap_pixels_after': int((coverage.target & ~after).sum()),
                   'selection_metric': 'head_depth_tested_pixel_centres_proxy_not_blender_coverage',
                   'existing_strands_preserved': True})
    # Existing trajectories are retained exactly; variable lengths use repeated endpoint padding.
    combined = existing + extra
    sizes = np.array([len(p) for p in combined])
    # Preserve even the input's sampling and padding, not just its polyline geometry.
    with np.load(args.input) as source:
        original = source['strands']
        sizes[:len(existing)] = source.get('valid_point_counts', np.full(len(existing), original.shape[1]))
    padded = pack_supplemental_strands(original, extra)
    print('Auditing final packed existing and supplemental segments before export', flush=True)
    result['combined_collision_audit'] = audit_strands(padded, mesh)
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2))
    if not result['combined_collision_audit']['passed']:
        raise RuntimeError('combined independent collision audit blocked export')
    np.savez_compressed(args.output_dir / 'all_root_prefixes.npz', strands=padded,
        valid_point_counts=sizes, template_blend_sha256=np.array(identity['template_sha256']),
        is_supplemental=np.arange(len(combined)) >= len(existing))
    np.savez_compressed(args.output_dir / 'view_evidence.npz', labels=labels, camera=camera,
                        selected_candidate_indices=proposal['indices'], selected_roots=proposal['points'])
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
