"""将发丝适配到 Blender 求值头模；保留根身份与全部非零前缀。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.compare_head_guard_integration import HeadSurface, measure
from scripts.recon_3d.recover_boundary_strands import export_all


def resample_polylines(strands, spacing=0.0005):
    """按弧长重采样，使用末点填充并保留每根发丝的有效点数。"""
    pieces = []
    for strand in strands:
        keep = np.r_[True, np.linalg.norm(np.diff(strand, axis=0), axis=1) > 1e-8]
        p = strand[keep]
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if len(p) > 1:
            t = np.linspace(0, length[-1], int(np.ceil(length[-1] / spacing)) + 1)
            p = np.column_stack([np.interp(t, length, p[:, axis]) for axis in range(3)])
        pieces.append(p)
    sizes = np.array([len(p) for p in pieces])
    result = np.empty((len(pieces), sizes.max(), 3), np.float32)
    for i, p in enumerate(pieces):
        result[i, :len(p)] = p
        result[i, len(p):] = p[-1]
    return result, sizes


def project_with_clearance(points, surface, clearance):
    """迭代沿最近表面法线向外修正，不修改输入数组。"""
    result = points.copy()
    for _ in range(8):
        distance, normal = surface.query(result)
        bad = distance < clearance
        if not bad.any():
            break
        result[bad] += (clearance - distance[bad] + 1e-6)[:, None] * normal[bad]
    return result


def fit(strands, surface, clearance=0.0005):
    """平滑修正位移，再投影以恢复中心线间隙；单位为米。"""
    original, sizes = resample_polylines(strands)
    fitted = original.copy()
    for iteration in range(8):
        fitted = project_with_clearance(
            fitted.reshape(-1, 3), surface, clearance
        ).reshape(original.shape)
        if iteration < 3:
            displacement = gaussian_filter1d(
                fitted - original, sigma=3, axis=1, mode="nearest"
            )
            fitted = original + displacement
        # 输出前缀的填充点必须跟随末端，不能形成新发丝段。
        for i, size in enumerate(sizes):
            fitted[i, size:] = fitted[i, size - 1]
    return fitted, original, sizes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--head',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path('results/multiview_data').resolve()):raise ValueError('输出必须位于多视图任务目录')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source=np.load(args.input);h=np.load(args.head)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(h['vertices']),o3d.utility.Vector3iVector(h['faces']))
    surface=HeadSurface(mesh);strands=source['strands']
    before=measure(strands,surface);print('before',before,flush=True)
    fitted,original,sizes=fit(strands,surface)
    after=measure(fitted,surface)
    # 独立加密采样验证整段管径余量，半径 0.3 mm，额外保留采样误差余量。
    if after['minimum_head_normal_distance_m']<.0004:raise RuntimeError('管径安全间隙未通过，不导出')
    payload={k:source[k] for k in source.files if k not in ['strands','termination_step']}
    payload['strands']=fitted;payload['valid_point_counts']=sizes
    if 'termination_step' in source:payload['source_termination_step']=source['termination_step']
    payload['source_roots_world']=strands[:,0]
    np.savez_compressed(args.output_dir/'all_root_prefixes.npz',**payload)
    export_all(args.output_dir/'all_root_prefixes.ply',fitted)
    offsets=np.linalg.norm(fitted-original,axis=2);root_offsets=np.linalg.norm(fitted[:,0]-strands[:,0],axis=1)
    report={'before':before,'after':after,'head_watertight':mesh.is_watertight(),
        'clearance_m':.0005,'render_bevel_radius_m':.0003,'audit_sample_spacing_m':.000125,
        'source':str(args.input.resolve()),'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'template_head':str(args.head.resolve()),'head_sha256':hashlib.sha256(args.head.read_bytes()).hexdigest(),
        'root_offset_quantiles_m':np.percentile(root_offsets,[0,50,90,100]).tolist(),
        'max_point_offset_m':float(offsets.max()),'root_identities_preserved':True,
        'root_coordinates_adjusted_for_template':True,'semantics':'render_template_adaptation_not_original_partition_solution',
        'original_partition_audit_applies':False,'selection':'all_nonzero_prefixes'}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
