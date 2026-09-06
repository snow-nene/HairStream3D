#!/usr/bin/env python3
"""固定根点、场和分区，对照逐步积分与头部几何约束。输出仅写指定任务目录。"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_segment_guard import first_invalid_segment_fraction, audit_volume_strands


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class HeadSurface:
    """局部最近三角形法线侧距离；开放网格不声称是闭合实体 SDF。"""
    def __init__(self, mesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def query(self, points):
        points = np.asarray(points, np.float32)
        result = self.scene.compute_closest_points(o3d.core.Tensor(points))
        normals = result['primitive_normals'].numpy()
        distance = np.sum((points - result['points'].numpy()) * normals, axis=1)
        return distance, normals

    def segment_minimum(self, starts, ends, sample_m):
        count = max(1, int(np.ceil(np.linalg.norm(ends-starts, axis=1).max(initial=0) / sample_m)))
        fractions = np.linspace(0, 1, count+1)
        points = starts[:, None] + fractions[None, :, None] * (ends-starts)[:, None]
        distance, _ = self.query(points.reshape(-1, 3))
        return distance.reshape(len(starts), -1).min(axis=1)


def independent_collision_audit(strands, mesh, sample_m=0.000125):
    """Check sampled strand points with closed-mesh ray occupancy."""
    if not mesh.is_watertight():
        raise ValueError("independent collision audit requires a watertight head mesh")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if sample_m <= 0:
        raise ValueError('sample spacing must be positive')
    inside, total = 0, 0
    for strand in np.asarray(strands):
        # Repeated terminal padding has no geometric extent. Keep the first
        # point so that a degenerate strand is still audited.
        keep = np.r_[True, np.linalg.norm(np.diff(strand, axis=0), axis=1) > 0]
        p = strand[keep]
        samples = [p[:1]]
        for start, end in zip(p[:-1], p[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / sample_m)))
            samples.append(start + np.linspace(0, 1, count + 1)[:, None] * (end - start))
        points = np.concatenate(samples, axis=0).astype(np.float32)
        # Multiple rays avoid parity errors at shared triangle edges/vertices.
        occupied = scene.compute_occupancy(o3d.core.Tensor(points), nsamples=11).numpy()
        inside += int(occupied.sum())
        total += len(points)
    return {"passed": inside == 0, "sampled_points": total,
            "inside_points": inside, "sample_spacing_m": sample_m, "occupancy_rays": 11}


def correct_direction(points, vectors, surface, traveled, clearance, ramp, band):
    distance, normals = surface.query(points)
    target = clearance * np.minimum((traveled + .003) / ramp, 1)
    tangent = vectors - np.sum(vectors * normals, axis=1)[:, None] * normals
    # 去除近表面向内分量，并用有限向外分量建立根部过渡；不移动根点。
    outward = np.maximum(np.sum(vectors*normals, axis=1), np.clip((target-distance)/.003, 0, 1))
    corrected = normalize(tangent + outward[:, None]*normals)
    return np.where((distance < band)[:, None], corrected, vectors)


def classify_contact(points, root_labels, labels, low, high, solid, conflict,
                     evidence_class=None, outer_excluded=None):
    """与 DDA 一致地检查接触面的全部相邻体素；保留同时出现的原因。"""
    shape = np.array(labels.shape)
    grid = (points-low)/(high-low)*(shape-1)
    near = np.abs(grid+.5-np.round(grid+.5)) < 2e-6
    base = np.floor(grid+.5).astype(int)
    base = np.where(near, np.round(grid+.5).astype(int), base)
    causes = [set() for _ in points]
    for bits in product((0, 1), repeat=3):
        offset = np.array(bits)
        enabled = ((offset == 0) | near).all(axis=1)
        ix = base-offset
        inside = ((ix >= 0)&(ix < shape)).all(axis=1)
        safe = np.clip(ix, 0, shape-1)
        for i in np.flatnonzero(enabled):
            if not inside[i]:
                causes[i].add('outside_grid')
                continue
            key = tuple(safe[i]); label = labels[key]
            if label == root_labels[i]:
                continue
            if solid[key]: causes[i].add('solid_voxel')
            elif conflict[key]: causes[i].add('conflict_voxel')
            elif label > 0: causes[i].add('partition_interface')
            elif outer_excluded is not None and outer_excluded[key]:
                causes[i].add('outer_envelope_exit')
            elif evidence_class is not None:
                causes[i].add({0: 'unknown_semantics', 1: 'unassigned_hair',
                               2: 'solid_semantics', 3: 'air_evidence_exit'}.get(
                                   int(evidence_class[key]), 'invalid_semantics'))
            else: causes[i].add('unlabeled_or_outside_domain')
    for i in range(len(points)):
        if np.any(points[i] <= low) or np.any(points[i] >= high): causes[i].add('outside_grid')
    return ['+'.join(sorted(c)) if c else 'unresolved_volume_contact' for c in causes]


def integrate(field, labels, roots, parts, low, high, solid, conflict, surface, mode, steps, step_m):
    n = len(roots); strands = np.repeat(roots[:, None], steps+1, axis=1).astype(np.float32)
    active = np.ones(n, bool); reasons = np.full(n, 'budget_exhausted', dtype='<U128')
    termination = np.full(n, steps, int); traveled = np.zeros(n); fallbacks = 0
    label_t = torch.as_tensor(labels.astype(np.int64)); shape = np.array(labels.shape)
    def query(p, identity, length):
        ix = np.rint((p-low)/(high-low)*(shape-1)).astype(int)
        ok = ((ix >= 0)&(ix < shape)).all(axis=1)
        ix = np.clip(ix, 0, shape-1); lab = labels[tuple(ix.T)]
        v = field[:, ix[:, 0], ix[:, 1], ix[:, 2]].T.copy()
        ok &= (lab == identity) & np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1)>1e-8)
        v = normalize(v)
        if mode == 'head_corrected': v = correct_direction(p, v, surface, length, .001, .006, .004)
        return v, ok
    for step in range(steps):
        strands[:, step+1] = strands[:, step]
        ids = np.flatnonzero(active)
        if not len(ids): break
        p = strands[ids, step].astype(float); identity = parts[ids]; length = traveled[ids]
        k1, ok1 = query(p, identity, length)
        k2, ok2 = query(p+step_m*.5*k1, identity, length+step_m*.5)
        k3, ok3 = query(p+step_m*.5*k2, identity, length+step_m*.5)
        k4, ok4 = query(p+step_m*k3, identity, length+step_m)
        delta = step_m*(k1+2*k2+2*k3+k4)/6
        fallback = ~(ok2 & ok3 & ok4); delta[fallback] = step_m*k1[fallback]
        fallbacks += int(fallback.sum())
        end = p+delta
        hit = first_invalid_segment_fraction(torch.as_tensor(p), torch.as_tensor(end), torch.as_tensor(identity), label_t, low, high).numpy()
        bad_volume = np.isfinite(hit)
        local_reason = np.full(len(ids), '', dtype='<U128')
        if bad_volume.any():
            contact = p[bad_volume]+hit[bad_volume, None]*delta[bad_volume]
            local_reason[bad_volume] = classify_contact(contact, identity[bad_volume], labels, low, high, solid, conflict)
        if mode != 'stepwise':
            # 与体积事件之前的几何比较；整步拒绝，避免输出未经复核的新连接。
            frac = np.where(bad_volume, np.clip(hit, 0, 1), 1)
            minimum = surface.segment_minimum(p, p+frac[:, None]*delta, .00025)
            head_bad = minimum < -.00001
            local_reason[head_bad] = 'head_surface_collision'
        local_reason[~ok1] = 'invalid_or_zero_direction'
        bad = local_reason != ''
        stop_ids = ids[bad]; active[stop_ids] = False
        reasons[stop_ids] = local_reason[bad]; termination[stop_ids] = step
        good = ids[~bad]; strands[good, step+1] = end[~bad]
        traveled[good] += np.linalg.norm(delta[~bad], axis=1)
        if step % 16 == 0: print(mode, step, 'active', int(active.sum()), flush=True)
    # 若全部提前终止，也必须把每根最后一点填充到预算末端。
    for i in range(n):
        if termination[i] < steps: strands[i, termination[i]+1:] = strands[i, termination[i]]
    return strands, reasons, termination, fallbacks


def measure(strands, surface):
    strands = np.asarray(strands)
    lengths = np.linalg.norm(np.diff(strands, axis=1), axis=2)
    minima = surface.query(strands.reshape(-1, 3))[0].reshape(strands.shape[:2]).min(axis=1)
    # 独立更密的线段采样（0.125 mm），包含停滞根点。
    for j in range(strands.shape[1]-1):
        minima = np.minimum(minima, surface.segment_minimum(strands[:, j], strands[:, j+1], .000125))
    if not np.isfinite(minima).all():
        raise ValueError('head distance query returned non-finite values')
    return {'roots': len(strands), 'length_quantiles_m': dict(zip(['min','p10','p25','median','p75','p90','max'], np.percentile(lengths.sum(1), [0,10,25,50,75,90,100]).tolist())),
            'zero_length_roots': int((lengths.sum(1)<1e-8).sum()), 'length_over_50mm_roots': int((lengths.sum(1)>.05).sum()),
            'head_inside_1mm_roots': int((minima<-.001).sum()), 'head_inside_3mm_roots': int((minima<-.003).sum()),
            'head_inside_10um_roots': int((minima<-.00001).sum()), 'minimum_head_normal_distance_m': float(minima.min())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['field','roots','bounds','bundle','baseline','head','output-dir']: ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(4)
    if args.output_dir.resolve().is_relative_to(Path('results/multiview_data').resolve()) is False: raise ValueError('输出必须位于 results/multiview_data/<image_id>/ 下')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    field_data = np.load(args.field); bundle = np.load(args.bundle); bounds = np.load(args.bounds); root_data = np.load(args.roots)
    field = field_data['field']; labels = field_data['partition_labels']; roots = root_data['roots_world']; parts = root_data['partition_labels']
    low, high = bounds['b_min'], bounds['b_max']; baseline = np.load(args.baseline)['strands']
    if not np.allclose(baseline[:,0], roots, atol=1e-7, rtol=0): raise ValueError('基线根点或顺序不同')
    ix = np.rint((roots-low)/(high-low)*(np.array(labels.shape)-1)).astype(int)
    if not np.array_equal(labels[tuple(ix.T)], parts): raise ValueError('根标签与场不一致')
    mesh = o3d.io.read_triangle_mesh(str(args.head)); surface = HeadSurface(mesh)
    manifest = {'inputs': {k: {'path': str(v.resolve()), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in vars(args).items() if k != 'output_dir'},
                'steps':64, 'step_m':.003, 'roots':len(roots), 'root_sha256':hashlib.sha256(roots.tobytes()).hexdigest(), 'head_watertight':mesh.is_watertight(),
                'distance_semantics':'nearest_triangle_normal_side_not_closed_sdf', 'head_tolerance_m':.00001, 'head_check_spacing_m':.00025, 'audit_spacing_m':.000125,
                'clearance_m':.001, 'root_ramp_m':.006, 'correction_band_m':.004,
                'policy':'same_roots_no_reseeding_no_partition_change_stop_before_first_unsafe_step; invalid_RK4_stage_falls_back_to_guarded_Euler',
                'termination_note':'budget_exhausted does not mean anatomically complete; head contact before volume contact takes precedence; simultaneous voxel causes retained'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    reports = {'historical_prefix': measure(baseline, surface)}
    print('baseline', reports, flush=True)
    for mode in ['stepwise','head_guard','head_corrected']:
        strands, reasons, termination, fallback = integrate(field, labels, roots, parts, low, high, bundle['solid_mask'], bundle['conflict'], surface, mode, 64, .003)
        np.savez_compressed(args.output_dir/(mode+'.npz'), strands=strands, root_labels=parts, source_indices=root_data['source_indices'], termination_reason=reasons, termination_step=termination)
        report = measure(strands, surface)
        report['termination_reasons'] = dict(zip(*[x.tolist() for x in np.unique(reasons, return_counts=True)]))
        report['rk4_euler_fallback_steps'] = fallback
        report['volume_audit'] = audit_volume_strands(strands, parts, labels, low, high)
        reports[mode] = report
        (args.output_dir/'comparison.json').write_text(json.dumps(reports, indent=2))
        print(mode, json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
