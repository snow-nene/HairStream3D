"""固定头模下分解旧轨迹、语义裁剪、重积分和补根的覆盖变化。"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, load_observation_camera, sample_polyline
from scripts.recon_3d.regrow_subject_strands import SubjectGuards


def unpack(path):
    with np.load(path) as d:
        return [s[np.r_[True, np.linalg.norm(np.diff(s, axis=0), axis=1)>0]].copy() for s in d['strands']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--analyze-tails', action='store_true')
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to(a.data_dir.resolve()):
        raise ValueError('输出须按图像聚合')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(a.run_dir/'geometry/template_head/template_head.npz') as d:
        v, f = d['vertices'], d['faces']
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    charts = {}
    for view in ['front', 'left', 'right', 'back']:
        seg = cv2.imread(str(a.data_dir/'maps/seg'/f'{view}.png'), 0)>127
        charts[view] = VisibleSurfaceGrowth(load_observation_camera(a.data_dir, view), seg.astype(int), np.zeros((*seg.shape, 2)), mesh)
    guards = SubjectGuards(charts)
    center = (v.min(0)+v.max(0))/2
    scale = 1.8/max(np.ptp(v[:,0]), np.ptp(v[:,2]))
    camera = np.array([[scale,0,0,-scale*center[0]], [0,0,scale,-scale*center[2]], [0,1,0,-center[1]], [0,0,0,1]])
    top = VisibleSurfaceGrowth(camera,np.ones((512,512),int),np.zeros((512,512,2)),mesh)
    top.target &= (top.head_points[:,1].reshape(512,512)>v[:,1].max()-.2*np.ptp(v[:,1])) & (top.normals[:,1].reshape(512,512)>.5)
    stages = {'v27': unpack(a.baseline)}
    clipped, reasons = [], {}
    tails = {'4': [], '5': [], '6': []}
    tail_records = []
    for source_id, s in enumerate(stages['v27']):
        samples = sample_polyline(s,.000125)
        codes = guards.points(samples,collision=False)
        bad = np.flatnonzero(codes)
        if len(bad):
            code = str(codes[bad[0]])
            reasons[code] = reasons.get(code,0)+1
            if a.analyze_tails:
                valid_tail = codes == 0
                valid_tail[:bad[0]+1] = False
                edges = np.diff(np.r_[False, valid_tail, False].astype(int))
                spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
                lengths, gaps, attach = [], [], []
                for start, end in spans:
                    piece = samples[start:end]
                    length = float(np.linalg.norm(np.diff(piece,axis=0),axis=1).sum())
                    if length < .004:
                        continue
                    tails[code].append(piece)
                    lengths.append(length)
                    gaps.append(float(np.linalg.norm(np.diff(samples[bad[0]:start+1],axis=0),axis=1).sum()))
                    near = charts['front'].scene.compute_closest_points(o3d.core.Tensor(piece[:1].astype(np.float32)))
                    root = near['points'].numpy()[0]+.0008*near['primitive_normals'].numpy()[0]
                    connector = sample_polyline(np.array([root,piece[0]]),.000125)
                    attach.append(bool(not guards.points(connector).any()))
                if lengths:
                    tail_records.append({'source_id':source_id,'first_stop':int(code),
                        'legal_tail_lengths_m':lengths,'arc_after_first_stop_m':gaps,
                        'straight_scalp_connector_passes':attach})
            samples = samples[:bad[0]]
            if len(samples)>1:
                clipped.append(samples)
        else:
            clipped.append(s)
    stages['only_semantic_clip'] = clipped
    stages['regrow'] = unpack(a.run_dir/'regrow/all_root_prefixes.npz')
    stages['final'] = unpack(a.run_dir/'final/all_root_prefixes.npz')
    report = {'clip_first_stop_counts':reasons,'views':{}}
    if a.analyze_tails:
        report['legal_tail_records'] = tail_records
        report['tail_interpretation'] = '诊断片段，未导出发丝；直线连接通过不代表方向或根点合理'
    for view, chart in {**charts,'top':top}.items():
        masks = {name:chart.covered(strands) & chart.target for name,strands in stages.items()}
        record = {}
        previous = None
        for name, mask in masks.items():
            record[name] = {'covered':int(mask.sum()),'gap':int((chart.target & ~mask).sum()),
                            'lost_from_v27':int((masks['v27'] & ~mask).sum())}
            if previous is not None:
                record[name]['lost_from_previous'] = int((previous & ~mask).sum())
                record[name]['gained_from_previous'] = int((mask & ~previous).sum())
            previous = mask
        report['views'][view] = record
        if a.analyze_tails:
            lost = masks['v27'] & ~masks['only_semantic_clip']
            final_gap = chart.target & ~masks['final']
            record['legal_tail_coverage'] = {}
            for code, pieces in tails.items():
                covered = chart.covered(pieces)
                record['legal_tail_coverage'][code] = {
                    'segments':len(pieces),'lost_pixels_overlapped':int((covered & lost).sum()),
                    'final_gap_pixels_overlapped':int((covered & final_gap).sum())}
        print(json.dumps({view:record}),flush=True)
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
