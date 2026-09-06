"""新鲜射线与头部遮挡下的双交点观测；来源及独立配准失败时只保留诊断。"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.mesh_observation_geometry import audit_registration,VisibleHairRaycaster,differential_visible_lift
from lib.multiview_observation import common_observation_gate,file_identity,verify_observation_manifest,source_budget_weights


def build_visible_observations(args):
    base,out=args.data_dir,args.output_dir
    expected=(base/'pde_governance/volume_partition_integration').resolve()
    if expected not in out.resolve().parents:raise ValueError('output must be in image integration directory')
    out.mkdir(parents=True,exist_ok=True)
    from scripts.recon_3d.recon3D import load_calib
    from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration
    manifest_path=base/'maps/observation_manifest.json'
    records=verify_observation_manifest(json.loads(manifest_path.read_text()),args.views) if manifest_path.is_file() else None
    report={'status':'building','config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'mesh':file_identity(args.hair_mesh),'head_prior':file_identity(args.head_mesh),'views':{},
            'provenance_manifest':file_identity(manifest_path) if records else None,'full_acceptance':False}
    confidence={};budget_records={}
    for view in args.views:
        strand_path=base/'maps/strand_map'/f'{view}.png';depth_path=base/'maps/depth_map'/f'{view}.npy'
        seg_path=base/'maps/seg'/f'{view}.png';partition_path=args.partition_dir/f'{view}_partition_evidence.npz'
        strand=cv2.cvtColor(cv2.imread(str(strand_path)),cv2.COLOR_BGR2RGB)
        depth=np.load(depth_path);seg=cv2.imread(str(seg_path),0)>127
        partition=np.load(partition_path);labels,edge=partition['labels'],partition['edge']
        if labels.shape!=seg.shape:raise ValueError('partition pixel grid mismatch')
        camera=(load_calib(str(base/'maps/param/front_dense_silhouette.npy')).numpy().astype(float)
                if view=='front' else load_blender_view_calibration(str(base),view)[0].astype(float))
        registration_path=args.registration_dir/f'{view}.npz' if args.registration_dir else None
        correspondences=np.load(registration_path) if registration_path and registration_path.is_file() else None
        registration=audit_registration(camera,seg.shape,correspondences,args.max_registration_error_px)
        y,x=np.mgrid[:seg.shape[0],:seg.shape[1]];pixels=np.c_[x.ravel(),y.ravel()]
        caster=VisibleHairRaycaster(args.hair_mesh,args.head_mesh,camera,seg.shape)
        surface=caster.cast_pixels(pixels)
        visible=surface['valid'].reshape(seg.shape)
        provisional=common_observation_gate(seg,strand,depth,visible,True)
        source_verified=bool(records is not None and (records[view]['source_kind']=='original'
                            or records[view].get('generation',{}).get('status')=='verified'))
        gate=common_observation_gate(seg,strand,depth,visible,registration['passed'] and source_verified)
        chosen=np.flatnonzero(provisional['accepted'].ravel())
        xy=pixels[chosen];rgb=strand.reshape(-1,3)[chosen].astype(float)/255
        directions=np.c_[1-2*rgb[:,2],2*rgb[:,1]-1]
        lift=differential_visible_lift(caster,xy,directions,labels,edge,args.delta_px,args.max_distance_m,args.max_angle_deg)
        final=lift['accepted']&registration['passed']&source_verified
        normal_valid=np.isfinite(lift['analytic_direction']).all(1)&(np.linalg.norm(lift['analytic_direction'],axis=1)>1e-8)
        summary={'registration':registration,'source_verified':source_verified,
                 'seg_pixels':int(seg.sum()),'mesh_visible_pixels':int(visible.sum()),
                 'head_occluded_in_seg':int((surface['head_occluded'].reshape(seg.shape)&seg).sum()),
                 'common_before_registration':len(chosen),'analytic_before_registration':int(normal_valid.sum()),
                 'differential_before_registration':int(lift['accepted'].sum()),'accepted':int(final.sum()),
                 'rejection_counts':{k:int(v.sum()) for k,v in lift.items() if k.endswith('rejected') or k in ['geometry_jump_or_zero','tangent_disagreement','invalid_direction']},
                 'source_assets':{str(p):file_identity(p) for p in [strand_path,depth_path,seg_path,partition_path]},
                 'registration_asset':file_identity(registration_path) if correspondences is not None else None}
        good_angles=lift['angle_degrees'][lift['accepted']]
        summary['accepted_provisional_angle_q90_degrees']=float(np.quantile(good_angles,.9)) if len(good_angles) else None
        np.savez_compressed(out/f'{view}_visible_observations.npz',**lift,strict_accepted=final,
                            pixel_xy=xy,camera=camera,provisional_only=np.asarray(not(registration['passed'] and source_verified)))
        np.savez_compressed(out/f'{view}_visibility_masks.npz',**gate,visible_hair=visible,
                            head_occluded=surface['head_occluded'].reshape(seg.shape))
        overlay=np.zeros((*seg.shape,3),np.uint8);overlay[seg]=[50,50,50]
        overlay[surface['head_occluded'].reshape(seg.shape)&seg]=[200,80,20]
        overlay.reshape(-1,3)[chosen]=[0,80,220]
        overlay.reshape(-1,3)[chosen[lift['accepted']]]=[80,200,80]
        cv2.imwrite(str(out/f'{view}_lifting_overlay.png'),overlay)
        confidence[view]=final.astype(float)
        budget_records[view]=records[view] if records else {'source_kind':'original' if view=='front' else 'generated',
                                                          'source_group':'original' if view=='front' else 'legacy_generated'}
        report['views'][view]=summary
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    weights=source_budget_weights(confidence,budget_records)
    np.savez_compressed(out/'source_weights.npz',**weights)
    report['status']='observations_ready' if all(r['registration']['passed'] and r['source_verified'] and r['accepted']>0 for r in report['views'].values()) else 'rejected_observation_contract'
    report['full_acceptance']=report['status']=='observations_ready'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({v:{k:r[k] for k in ['head_occluded_in_seg','common_before_registration','differential_before_registration','accepted','accepted_provisional_angle_q90_degrees']} for v,r in report['views'].items()},indent=2))
    print(report['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--partition-dir',type=Path,required=True)
    parser.add_argument('--registration-dir',type=Path)
    parser.add_argument('--hair-mesh',type=Path,required=True)
    parser.add_argument('--head-mesh',type=Path,default=Path('data/head_model.obj'))
    parser.add_argument('--views',nargs='+',default=['front'])
    parser.add_argument('--delta-px',type=float,default=2.)
    parser.add_argument('--max-distance-m',type=float,default=.008)
    parser.add_argument('--max-angle-deg',type=float,default=20.)
    parser.add_argument('--max-registration-error-px',type=float,default=8.)
    build_visible_observations(parser.parse_args())
