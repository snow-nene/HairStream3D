#!/usr/bin/env python3
"""同分区连续插值、受限转向和头部守卫；保留全部根身份并导出全部有效前缀。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.compare_head_guard_integration import HeadSurface, normalize, measure, classify_contact
from lib.recon_strategy.volume_segment_guard import first_invalid_segment_fraction, audit_volume_strands


def sample_partition_field(points, identity, field, labels, low, high):
    """只用同区非零邻点做三线性插值，禁止混合对岸向量。"""
    shape = np.array(labels.shape); g = (points-low)/(high-low)*(shape-1)
    base = np.floor(g).astype(int); frac = g-base
    values = np.zeros_like(points, dtype=float); total = np.zeros(len(points))
    for bit in np.ndindex(2,2,2):
        bit = np.array(bit); ix = base+bit
        inside = ((ix>=0)&(ix<shape)).all(1); safe = np.clip(ix,0,shape-1)
        v = field[:,safe[:,0],safe[:,1],safe[:,2]].T
        weight = np.prod(np.where(bit,frac,1-frac),axis=1)
        weight *= inside & (labels[tuple(safe.T)]==identity) & (np.linalg.norm(v,axis=1)>1e-8)
        values += weight[:,None]*v; total += weight
    return normalize(values), (total>1e-8)&(np.linalg.norm(values,axis=1)>1e-8)


def head_direction(points, vectors, head, step, traveled):
    d,n = head.query(points); target = .001*np.minimum((traveled+step)/.006,1)
    vn = np.sum(vectors*n,axis=1); tangent = vectors-vn[:,None]*n
    outward = np.maximum(vn,np.clip((target-d)/np.maximum(step,1e-8),0,1))
    return np.where((d<.004)[:,None],normalize(tangent+outward[:,None]*n),vectors)


def cone_directions(reference):
    """在原场方向的 60 度锥内搜索；只在候选整段失败后使用。"""
    axis = np.tile([0.,0.,1.],(len(reference),1))
    axis[np.abs(reference[:,2])>.9] = [0.,1.,0.]
    u = normalize(np.cross(reference,axis)); v = np.cross(reference,u)
    options = [reference]
    for degree in (15,30,45,60):
        angle = np.deg2rad(degree)
        for phi in np.arange(8)*np.pi/4:
            options.append(np.cos(angle)*reference+np.sin(angle)*(np.cos(phi)*u+np.sin(phi)*v))
    return np.stack(options,axis=1)


def grow(field, labels, roots, parts, low, high, bundle, head, max_steps=256, budget=.192, turn_limit=60):
    n = len(roots); points = np.repeat(roots[:,None],max_steps+1,axis=1).astype(np.float32)
    active = np.ones(n,bool); length = np.zeros(n); previous = np.zeros((n,3)); steps = np.zeros(n,int)
    reasons = np.full(n,'step_limit',dtype='<U128'); recovery = np.zeros(n,int); max_deviation = np.zeros(n)
    labels_t = torch.as_tensor(labels.astype(np.int64)); initial_reference,_ = sample_partition_field(roots,parts,field,labels,low,high)
    first_block = np.full(n,'',dtype='<U128')
    for step_index in range(max_steps):
        points[:,step_index+1] = points[:,step_index]
        ids = np.flatnonzero(active)
        if not len(ids): break
        p = points[ids,step_index].astype(float); identity = parts[ids]
        h = np.minimum(.003,budget-length[ids]); ref, supported = sample_partition_field(p,identity,field,labels,low,high)
        ref = head_direction(p,ref,head,h,length[ids]); direction = ref
        # 中点积分；中点非法不会被直接接受，最终仍做全段 DDA 和几何审计。
        midpoint = p+.5*h[:,None]*ref
        mid,mid_ok = sample_partition_field(midpoint,identity,field,labels,low,high)
        mid = head_direction(midpoint,mid,head,h,length[ids]+.5*h)
        direction = np.where(mid_ok[:,None],mid,ref)
        candidate = (p+h[:,None]*direction).astype(np.float32).astype(float)
        hit = first_invalid_segment_fraction(torch.as_tensor(p),torch.as_tensor(candidate),torch.as_tensor(identity),labels_t,low,high).numpy()
        head_min = head.segment_minimum(p,candidate,.000125)
        has_previous = np.linalg.norm(previous[ids],axis=1)>.1
        turn_ok = ~has_previous | (np.sum(direction*previous[ids],axis=1)>=np.cos(np.deg2rad(45)))
        reference_ok = np.sum(direction*ref,axis=1)>=np.cos(np.deg2rad(turn_limit))
        ok = np.isinf(hit)&(head_min>=-.000005)&supported&turn_ok&reference_ok
        reason = np.full(len(ids),'turn_limit',dtype='<U128')
        volume_bad = np.isfinite(hit)
        if volume_bad.any():
            reason[volume_bad] = classify_contact(
                p[volume_bad]+hit[volume_bad,None]*(candidate-p)[volume_bad],
                identity[volume_bad], labels, low, high,
                bundle['solid_mask'], bundle['conflict'],
                evidence_class=bundle.get('evidence_class'),
                outer_excluded=bundle.get('outer_excluded'))
        reason[head_min<-.000005] = 'head_surface_collision'
        reason[~supported] = 'unsupported_direction'
        unset = first_block[ids]==''; first_block[ids[unset&~ok]] = reason[unset&~ok]
        blocked = np.flatnonzero(~ok & supported)
        if len(blocked):
            cone = cone_directions(ref[blocked]); count = cone.shape[1]
            # 对同一根只接受最小必要偏角；依次尝试原步长、半步、四分之一步。
            remaining = np.arange(len(blocked))
            for factor in (1.,.5,.25):
                if not len(remaining): break
                loc = blocked[remaining]; choices = cone[remaining]
                hs = h[loc]*factor; ends = (p[loc,None]+hs[:,None,None]*choices).astype(np.float32).astype(float)
                origins = np.repeat(p[loc],count,axis=0); endpoints = ends.reshape(-1,3)
                hits = first_invalid_segment_fraction(torch.as_tensor(origins),torch.as_tensor(endpoints),torch.as_tensor(np.repeat(identity[loc],count)),labels_t,low,high).numpy().reshape(-1,count)
                valid = np.isinf(hits)
                angle_allowed = np.sum(choices*ref[loc,None],axis=2)>=np.cos(np.deg2rad(turn_limit))-1e-7
                prev_allowed = (~has_previous[loc,None]) | (np.sum(choices*previous[ids[loc],None],axis=2)>=np.cos(np.deg2rad(45)))
                valid &= angle_allowed & prev_allowed
                flat = np.flatnonzero(valid)
                if len(flat):
                    mins = head.segment_minimum(origins[flat],endpoints[flat],.000125)
                    valid.flat[flat] &= mins>=-.000005
                # 禁止返回历史点附近形成环；窗口之外 1.5 mm 以内视为复访。
                for row,l in enumerate(loc):
                    if step_index>=12:
                        history = points[ids[l],:step_index-8]
                        proximity = np.linalg.norm(ends[row,:,None]-history[None],axis=2).min(1)
                        valid[row] &= proximity>.0015
                found = valid.any(1); choice = valid.argmax(1)
                rows = np.flatnonzero(found); selected = loc[found]
                candidate[selected] = ends[rows,choice[found]]; direction[selected] = choices[rows,choice[found]]
                ok[selected] = True; recovery[ids[selected]] += 1
                remaining = remaining[~found]
        ok &= supported
        # 正常方向也做历史复访检查，避免只有回退路径受约束。
        for local in np.flatnonzero(ok):
            if step_index>=12 and np.linalg.norm(points[ids[local],:step_index-8]-candidate[local],axis=1).min()<.0015:
                ok[local]=False; reason[local]='loop_guard'
        stopped = ids[~ok]; active[stopped] = False; reasons[stopped] = reason[~ok]
        good = ids[ok]; displacement = candidate[ok]-p[ok]
        points[good,step_index+1] = candidate[ok]; steps[good] = step_index+1
        length[good] += np.linalg.norm(displacement,axis=1); previous[good] = normalize(displacement)
        angle = np.degrees(np.arccos(np.clip(np.sum(normalize(displacement)*ref[ok],axis=1),-1,1)))
        max_deviation[good] = np.maximum(max_deviation[good],angle)
        finished = good[length[good]>=budget-1e-6]; reasons[finished]='budget_exhausted';active[finished]=False
        if step_index%16==0: print('step',step_index,'active',int(active.sum()),'finished',int((reasons=='budget_exhausted').sum()),flush=True)
    for i in range(n): points[i,steps[i]+1:]=points[i,steps[i]]
    return points,reasons,steps,recovery,max_deviation,first_block


def export_all(path, strands):
    """导出所有有非零段的根前缀，移除存储填充点；身份保存在伴随 NPZ。"""
    vertices=[];edges=[];offset=0
    for strand in strands:
        keep=np.r_[True,np.linalg.norm(np.diff(strand,axis=0),axis=1)>1e-8]
        s=strand[keep]
        if len(s)<2:continue
        vertices.append(s);edges.append(np.column_stack([np.arange(offset,offset+len(s)-1),np.arange(offset+1,offset+len(s))]));offset+=len(s)
    lines=o3d.geometry.LineSet();lines.points=o3d.utility.Vector3dVector(np.concatenate(vertices));lines.lines=o3d.utility.Vector2iVector(np.concatenate(edges))
    o3d.io.write_line_set(str(path),lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(4)
    if not a.output_dir.resolve().is_relative_to(a.base.resolve()):raise ValueError('输出须位于该 Image ID 的任务目录')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    paths={'field':a.base/'mesh_front_128_partition_interface_repaired_iter4/repaired_partition_field.npz','roots':a.base/'mesh_front_128_partition_interface_repaired_iter4/rebound_roots.npz','bundle':a.base/'mesh_front_128_downgraded/trusted_partition_bundle_128.npz','bounds':a.base/'mesh_front_64/candidate_domain_seeds.npz','head':Path('data/head_model.obj'),'baseline':a.base/'head_guard_comparison_20260906/head_corrected.npz'}
    f=np.load(paths['field']);r=np.load(paths['roots']);b=np.load(paths['bundle']);bounds=np.load(paths['bounds']);head=HeadSurface(o3d.io.read_triangle_mesh(str(paths['head'])))
    manifest={'inputs':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()}for k,v in paths.items()},'policy':'same_root_partition_continuous_interpolation_bounded_turn_no_domain_edit','step_m':.003,'minimum_step_m':.00075,'budget_m':.192,'root_count':len(r['roots_world']),'maximum_search_angle_deg':60,'previous_direction_limit_deg':45,'head_check_m':.000125,'head_tolerance_m':.000005,'root_preserved':True}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    strands,reasons,steps,recovery,angles,first_block=grow(f['field'],f['partition_labels'],r['roots_world'],r['partition_labels'],bounds['b_min'],bounds['b_max'],b,head)
    np.savez_compressed(a.output_dir/'all_root_prefixes.npz',strands=strands,root_labels=r['partition_labels'],source_indices=r['source_indices'],termination_reason=reasons,termination_step=steps,recovery_steps=recovery,maximum_deviation_deg=angles,first_block_reason=first_block)
    report=measure(strands,head);report['termination_reasons']=dict(zip(*[x.tolist()for x in np.unique(reasons,return_counts=True)]));report['recovered_roots']=int((recovery>0).sum());report['maximum_deviation_deg']=float(angles.max());report['volume_audit']=audit_volume_strands(strands,r['partition_labels'],f['partition_labels'],bounds['b_min'],bounds['b_max']);report['render_selection']='all_nonzero_prefixes_regardless_of_termination';report['exported_roots']=int((np.linalg.norm(np.diff(strands,axis=1),axis=2).sum(1)>1e-8).sum())
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['volume_audit']['passed'] or report['head_inside_10um_roots']:
        raise RuntimeError('最终几何审计未通过，保留诊断但拒绝 PLY 导出')
    export_all(a.output_dir/'all_root_prefixes.ply',strands)
    export_all(a.output_dir/'baseline_all_prefixes.ply',np.load(paths['baseline'])['strands'])

if __name__=='__main__':main()
