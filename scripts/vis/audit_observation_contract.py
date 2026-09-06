"""在已有网格交点缓存上审计观测契约；不为历史数据补造来源或配准证明。"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.multiview_observation import (
    common_observation_gate,source_budget_weights,file_identity,verify_observation_manifest,
)
from lib.multiview_registration import registration_report


def audit_cached_observations(args):
    base,source,out=args.data_dir,args.seed_dir,args.output_dir
    expected=(base/'pde_governance/volume_partition_integration').resolve()
    if expected not in out.resolve().parents or source.resolve()==out.resolve():
        raise ValueError('output must be a separate image integration run')
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=base/'maps/observation_manifest.json'
    provenance=None
    if manifest_path.is_file():
        provenance=verify_observation_manifest(json.loads(manifest_path.read_text()),args.views)
    original=file_identity(base/'raw_img.png')
    report={'status':'diagnostic_only','provenance_verified':provenance is not None,
            'calibration_verified':False,'calibration_reason':'cached ray intersections do not prove image-to-mesh registration',
            'full_acceptance':False,'original_front':original,'views':{},
            'mesh_support_scope':'cached hair mesh hits with silhouette margin and incidence filter; no new head occlusion check',
            'weight_scope':'provisional sampling masses only; not applied to PDE'}
    confidence={};records={}
    for view in args.views:
        paths={k:base/'maps'/k/f'{view}.{suffix}' for k,suffix in
               [('strand_map','png'),('depth_map','npy'),('seg','png')]}
        strand=cv2.cvtColor(cv2.imread(str(paths['strand_map'])),cv2.COLOR_BGR2RGB)
        seg=cv2.imread(str(paths['seg']),0)>127;depth=np.load(paths['depth_map'])
        cache_path=source/f'{view}_calibration_samples.npz';cache=np.load(cache_path)
        mesh=np.zeros(seg.shape,bool);mesh[cache['pixel_y'],cache['pixel_x']]=True
        pixel_xy=np.column_stack([cache['pixel_x'],cache['pixel_y']])
        pixel_camera=np.array([[0.5*(seg.shape[1]-1),0,0,0.5*(seg.shape[1]-1)],
                               [0,0.5*(seg.shape[0]-1),0,0.5*(seg.shape[0]-1)],
                               [0,0,1,0],[0,0,0,1]],dtype=float) @ cache['camera']
        registration=registration_report(cache['mesh_world'],pixel_xy,pixel_camera,seg.shape,
                                         max_error_px=args.max_reprojection_px)
        strict=common_observation_gate(seg,strand,depth,mesh,registration['passed'])
        # This map deliberately separates pre-calibration eligibility from the
        # actual (zero) accepted mask. It must not be consumed as a PDE gate.
        provisional=common_observation_gate(seg,strand,depth,mesh,True)['accepted']
        np.savez_compressed(out/f'{view}_observation_masks.npz',**strict,
                            provisional_before_calibration=provisional)
        confidence[view]=provisional.astype(float)
        records[view]=provenance[view] if provenance else {
            'source_kind':'original' if view=='front' else 'generated',
            'source_group':('original:' if view=='front' else 'flux:')+original['sha256']}
        report['views'][view]={'counts':{k:int(v.sum()) for k,v in strict.items()},
            'provisional_before_calibration':int(provisional.sum()),
            'registration':registration,
            'source_identity_status':'recorded' if provenance else 'legacy_expected_role_not_verified',
            'assets':{k:file_identity(p) for k,p in paths.items()},'cache':file_identity(cache_path)}
        overlay=np.zeros((*seg.shape,3),np.uint8);overlay[seg]=[50,50,50]
        overlay[strict['mesh_not_visible']]=[0,100,255]
        overlay[strict['invalid_depth']|strict['invalid_strand']]=[0,0,255]
        overlay[provisional]=[80,180,80]
        cv2.imwrite(str(out/f'{view}_provisional_overlay.png'),overlay)
    weights=source_budget_weights(confidence,records)
    np.savez_compressed(out/'provisional_source_weights.npz',**weights)
    report['provisional_weight_mass']={view:float(w.sum()) for view,w in weights.items()}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['views','original_front']},ensure_ascii=False,indent=2))
    print(json.dumps({v:{'seg':int(np.load(out/f'{v}_observation_masks.npz')['outside_seg'].size-r['counts']['outside_seg']),
                        'provisional':r['provisional_before_calibration'],'accepted':r['counts']['accepted']} for v,r in report['views'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--seed-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--views',nargs='+',default=['front'])
    parser.add_argument('--max-reprojection-px',type=float,default=1.0)
    audit_cached_observations(parser.parse_args())
