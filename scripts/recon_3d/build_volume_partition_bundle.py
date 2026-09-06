"""将已审计体积证据与可见 strand/depth 分区组装成统一输入包。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_partition import (
    adapt_refined_evidence, surface_partition_seeds, propagate_partitions, save_bundle,
    merge_view_partition_seeds,
)
from lib.multiview_registration import registration_report
from lib.multiview_direction_lifting import lift_visible_directions, compare_axial_directions
from scripts.vis.audit_strand_depth_partitions import (
    compute_partition_evidence, clean_candidate_edge, partition_from_barrier,
    select_topological_edge_components,
)


def visible_mesh_samples(mesh_path, camera, seg, return_normals=False):
    """First visible mesh intersections in world metres, with optional normals."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    y, x = np.where(ndimage.distance_transform_edt(seg) >= 5)
    uv = np.column_stack([2*x/(seg.shape[1]-1)-1, 2*y/(seg.shape[0]-1)-1])
    clip = np.column_stack([np.asarray(mesh.vertices), np.ones(len(mesh.vertices))]) @ camera.T
    near_z = clip[:, 2].max() + 1
    inverse = np.linalg.inv(camera)
    near = np.column_stack([uv, np.full(len(uv), near_z), np.ones(len(uv))]) @ inverse.T
    far = np.column_stack([uv, np.full(len(uv), near_z-1), np.ones(len(uv))]) @ inverse.T
    near, far = near[:, :3]/near[:, 3:4], far[:, :3]/far[:, 3:4]
    direction = far-near
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    hit = scene.cast_rays(o3d.core.Tensor(np.column_stack([near,direction]).astype(np.float32)))
    distance = hit["t_hit"].numpy()
    incidence = np.abs((hit["primitive_normals"].numpy()*direction).sum(1))
    valid = np.isfinite(distance) & (incidence > .4)
    result = (y[valid], x[valid], near[valid] + direction[valid]*distance[valid,None])
    return (*result, hit["primitive_normals"].numpy()[valid]) if return_normals else result


def calibrate_visible_depth(depth, camera, y, x, world, tolerance_m):
    """Robust affine neural-depth calibration with a spatial holdout gate."""
    predicted = depth[y,x]
    target = (np.column_stack([world,np.ones(len(world))]) @ camera.T)[:,2]
    valid = np.isfinite(predicted) & (predicted > .05)
    holdout = ((x//16 + y//16) % 5 == 0) & valid
    train = valid & ~holdout
    if train.sum() < 30 or holdout.sum() < 10 or np.std(predicted[train]) < 1e-6:
        raise ValueError("Insufficient nonconstant neural depth for calibration")
    design = np.column_stack([predicted[train],np.ones(train.sum())])
    weight = np.ones(train.sum())
    for _ in range(8):
        coefficients = np.linalg.lstsq(design*np.sqrt(weight[:,None]),target[train]*np.sqrt(weight),rcond=None)[0]
        residual = design@coefficients-target[train]
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))),1e-6)
        weight = np.minimum(1,1.345*scale/np.maximum(np.abs(residual),1e-12))
    meters_per_z = np.linalg.norm(np.linalg.inv(camera)[:3,2])
    error_m = np.abs(predicted*coefficients[0]+coefficients[1]-target)*meters_per_z
    q90 = float(np.quantile(error_m[holdout],.9))
    report = {"affine_scale_offset":coefficients.tolist(),"meters_per_camera_z":float(meters_per_z),
              "holdout_points":int(holdout.sum()),"holdout_q90_m":q90,
              "tolerance_m":float(tolerance_m),"passed":q90<=tolerance_m}
    return coefficients, valid & (error_m<=tolerance_m), report


def build_bundle(args):
    base, out = args.data_dir, args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    parent = base/'pde_governance/volume_domain_audit'
    arrays, meta = adapt_refined_evidence(
        parent/'step_07_surface_evidence_continuity/refined_evidence.npz',
        parent/'step_06_four_class_evidence/volume_evidence.npz',
        parent/'step_05_closed_external_envelope/closed_outer_candidate.npz')
    domain, spacing, origin = arrays['domain_mask'],np.array(meta['spacing']),np.array(meta['origin'])
    from scripts.recon_3d.recon3D import load_calib
    from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration
    global_seeds = np.zeros(domain.shape,np.int32)
    global_directions = np.zeros((*domain.shape,3),np.float32)
    report = {"views":{},"domain_voxels":int(domain.sum()),"status":"building"}
    np.savez_compressed(out/'candidate_volume.npz', **arrays,
                        metadata_json=np.asarray(json.dumps(meta)))
    # Diagnostic evidence is intentionally not a valid PDE bundle until all
    # calibration, seed and component gates pass.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for axis, plot in enumerate(axes):
        plot.imshow(np.take(domain,domain.shape[axis]//2,axis=axis).T,origin='lower')
        plot.set_title(f'Candidate domain: axis {axis}')
    fig.tight_layout();fig.savefig(out/'domain_sections.png',dpi=120);plt.close(fig)
    source_paths = [Path(p) for p in meta['sources']]
    for view in args.views:
        strand_path, depth_path, seg_path = (base/'maps'/kind/f'{view}.{suffix}' for kind,suffix in [('strand_map','png'),('depth_map','npy'),('seg','png')])
        source_paths.extend([strand_path,depth_path,seg_path])
        strand = cv2.cvtColor(cv2.imread(str(strand_path)),cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        seg = cv2.imread(str(seg_path),0)>127
        if seg.shape != depth.shape or strand.shape[:2] != depth.shape:
            raise ValueError(f"{view}: maps must share calibrated pixel grid")
        evidence = compute_partition_evidence(strand,depth,seg,reject_depth_planes=True)
        threshold_audit = {}
        for quantile in (.85,.90,.93,.96):
            candidate_edge, candidate_threshold = clean_candidate_edge(evidence['fused'],evidence['valid'],quantile)
            candidate_labels, candidate_report = partition_from_barrier(seg,candidate_edge)
            threshold_audit[str(quantile)] = dict(candidate_report, threshold=candidate_threshold)
            np.save(out/f'{view}_q{round(quantile*100)}_labels.npy',candidate_labels)
        edge, threshold = clean_candidate_edge(evidence['fused'],evidence['valid'],args.edge_quantile)
        # Existing topology filter uses a five-pixel support band to bridge the
        # explicitly suppressed silhouette margin; it adds no free-form cut.
        edge, topology_report = select_topological_edge_components(
            seg,edge,barrier_radius=5,min_region_pixels=max(256,int(seg.sum()*.05)))
        labels, partition_report = partition_from_barrier(
            seg,edge,barrier_radius=5,min_region_pixels=max(256,int(seg.sum()*.05)))
        partition_report['topology_filter']=topology_report
        np.savez_compressed(out/f'{view}_partition_evidence.npz',**evidence,labels=labels,edge=edge)
        colors = np.random.default_rng(42).integers(40,240,size=(int(labels.max())+1,3),dtype=np.uint8)
        colors[0]=0
        cv2.imwrite(str(out/f'{view}_partitions.png'),colors[labels])
        raw_path = base/'raw_img.png' if view=='front' else base/'maps/body_img'/f'{view}.png'
        raw = cv2.imread(str(raw_path))
        if raw is not None:
            raw=cv2.resize(raw,(seg.shape[1],seg.shape[0]))
            overlay=cv2.addWeighted(raw,.65,colors[labels],.35,0)
            overlay[edge]=(0,0,255)
            cv2.imwrite(str(out/f'{view}_partition_overlay.png'),overlay)
        if view=='front':
            calib_path=base/'maps/param/front_dense_silhouette.npy'
            camera=load_calib(str(calib_path)).numpy().astype(float)
            source_paths.append(calib_path)
        else:
            camera=load_blender_view_calibration(str(base),view)[0].astype(float)
        seed_geometry = getattr(args, 'seed_geometry', 'neural_depth')
        y,x,mesh_world,mesh_normals=visible_mesh_samples(args.mesh,camera,seg,return_normals=True)
        pixel_camera = np.array([[0.5*(seg.shape[1]-1), 0, 0, 0.5*(seg.shape[1]-1)],
                                 [0, 0.5*(seg.shape[0]-1), 0, 0.5*(seg.shape[0]-1)],
                                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float) @ camera
        registration = registration_report(
            mesh_world, np.column_stack([x, y]), pixel_camera, seg.shape,
            max_error_px=args.max_reprojection_px,
        )
        if not registration['passed']:
            report['views'][view] = {'registration': registration,
                                     'status': 'rejected_registration'}
            (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            raise ValueError(f'{view}: camera/mesh registration gate failed')
        if seed_geometry == 'mesh_raycast':
            world = mesh_world
            usable = np.ones(len(world), dtype=bool)
            calibration = {'required': False, 'passed': None,
                           'reason': 'mesh_raycast geometry; neural depth used only for partition evidence'}
        else:
            coeff,usable,calibration=calibrate_visible_depth(depth,camera,y,x,mesh_world,args.depth_tolerance_m)
        report['views'][view]={'threshold':threshold,'partition':partition_report,
                              'threshold_audit':threshold_audit,'calibration':calibration}
        report['views'][view]['registration'] = registration
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        report['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        np.savez_compressed(out/f'{view}_calibration_samples.npz',pixel_y=y,pixel_x=x,
                            mesh_world=mesh_world,neural_depth=depth[y,x],camera=camera,
                            accepted=usable)
        if seed_geometry == 'neural_depth' and not calibration['passed']:
            report['status']='rejected_depth_calibration'
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if seed_geometry == 'neural_depth' and not calibration['passed']:
            raise ValueError(f"{view}: depth calibration holdout failed; see report.json")
        if seed_geometry == 'neural_depth':
            uv=np.column_stack([2*x/(seg.shape[1]-1)-1,2*y/(seg.shape[0]-1)-1])
            projected=np.column_stack([uv,depth[y,x]*coeff[0]+coeff[1],np.ones(len(x))]) @ np.linalg.inv(camera).T
            world=projected[:,:3]/projected[:,3:4]
        usable &= labels[y,x]>0
        local=surface_partition_seeds(domain,origin,spacing,world[usable],labels[y[usable],x[usable]],args.surface_band_m)
        # Lift the observed line tangent through the calibrated depth surface.
        from scipy.spatial import cKDTree
        strand_unit=strand.astype(float)/255
        dx=1-2*strand_unit[y[usable],x[usable],2]
        dy=2*strand_unit[y[usable],x[usable],1]-1
        vx=dx*2/(seg.shape[1]-1)
        vy=dy*2/(seg.shape[0]-1)
        inverse_linear = np.linalg.inv(camera)[:3,:3]
        if seed_geometry == 'mesh_raycast':
            # Preserve the observed image tangent while enforcing n dot t = 0.
            # Camera-z motion lies on the same image ray, so this correction
            # does not change the projected strand direction.
            base_tangent=np.column_stack([vx,vy,np.zeros_like(vx)])@inverse_linear.T
            ray=inverse_linear[:,2]
            normals=mesh_normals[usable]
            denominator=normals@ray
            if np.any(np.abs(denominator)<1e-8):
                raise ValueError('Degenerate mesh tangent despite incidence gate')
            vz=-np.sum(normals*base_tangent,axis=1)/denominator
        else:
            smooth_depth=np.where(np.isfinite(depth),depth,0)*coeff[0]+coeff[1]
            dzdy,dzdx=np.gradient(smooth_depth)
            vz=dzdx[y[usable],x[usable]]*dx+dzdy[y[usable],x[usable]]*dy
        tangents=np.column_stack([vx,vy,vz])@inverse_linear.T
        tangents/=np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-12)
        if seed_geometry == 'mesh_raycast':
            lifted = lift_visible_directions(
                y[usable], x[usable], world[usable], labels[y[usable], x[usable]],
                strand, seg.shape, step_px=args.lifting_step_px,
            )
            # Differential lifting is preferred when the adjacent visible
            # sample is continuous. Mesh tangent remains a lower-confidence
            # fallback for isolated pixels; it never bridges a partition.
            lifted_world = lifted['directions'].astype(float)
            lifted_world = lifted_world @ np.eye(3)
            consistency = compare_axial_directions(
                lifted_world, tangents, lifted['accepted'], args.max_lifting_angle_deg)
            tangents[consistency['compatible']] = lifted_world[consistency['compatible']]
            report['views'][view]['direction_lifting'] = {
                'step_px': args.lifting_step_px,
                'accepted': int(lifted['accepted'].sum()),
                'rejected': int((~lifted['accepted']).sum()),
                'reasons': {str(k): int(v) for k, v in zip(*np.unique(lifted['reason'], return_counts=True))},
                'consistency': {
                    'max_angle_deg': args.max_lifting_angle_deg,
                    'usable': int(consistency['usable'].sum()),
                    'compatible': int(consistency['compatible'].sum()),
                    'incompatible': int((consistency['usable'] & ~consistency['compatible']).sum()),
                    'median_angle_deg': float(np.median(consistency['angle_deg'][consistency['usable']]))
                    if consistency['usable'].any() else None,
                },
            }
        seed_indices=np.argwhere(local>0)
        _,nearest=cKDTree(world[usable]).query(origin+seed_indices*spacing)
        local_directions=np.zeros_like(global_directions)
        local_directions[tuple(seed_indices.T)]=tangents[nearest]
        global_seeds,global_directions,conflict,matches=merge_view_partition_seeds(
            global_seeds,global_directions,local,local_directions,args.min_overlap_voxels)
        arrays['conflict'] |= conflict
        report['views'][view]['global_correspondence']=matches
        report['views'][view]['seed_geometry']=seed_geometry
        report['views'][view]['surface_samples']=int(usable.sum())
        report['views'][view]['seed_voxels']=int((local>0).sum())
        np.savez_compressed(out/f'{view}_surface_seeds.npz',world=world[usable],
                            labels=labels[y[usable],x[usable]],pixel_y=y[usable],pixel_x=x[usable],
                            tangents=tangents,seed_volume=local,camera=camera)
    components,n=ndimage.label(domain)
    anchored=np.unique(components[global_seeds>0])
    unseeded=domain & ~np.isin(components,anchored)
    report['components']=int(n)
    report['unseeded_voxels']=int(unseeded.sum())
    np.savez_compressed(out/'candidate_domain_seeds.npz',**arrays,seeds=global_seeds,
                        seed_directions=global_directions,b_min=origin,b_max=origin+spacing*(np.array(domain.shape)-1))
    if unseeded.any():
        report['status']='rejected_unseeded_components'
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        raise ValueError(f"{unseeded.sum()} unseeded domain voxels; no silent pruning")
    labels,confidence=propagate_partitions(domain,global_seeds,spacing,arrays['evidence_confidence'])
    arrays['partition_labels'],arrays['partition_confidence']=labels,confidence
    meta['seed_geometry']=seed_geometry
    meta['depth_convention']=('mesh first visible hit in world metres; neural depth only partitions'
                              if seed_geometry=='mesh_raycast' else
                              'affine neural depth to calibrated orthographic camera_z; metric surface band')
    arrays['seed_labels']=global_seeds
    arrays['seed_directions']=global_directions
    meta['views']=args.views
    source_paths.append(args.mesh)
    meta['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    meta['inferred_interior']=True
    save_bundle(out/'volume_partition_bundle.npz',arrays,meta)
    report['status']='bundle_ready'
    report['partition_counts']={str(int(k)):int(v) for k,v in zip(*np.unique(labels[domain],return_counts=True))}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--views',nargs='+',default=['front'])
    parser.add_argument('--seed-geometry',choices=['neural_depth','mesh_raycast'],default='neural_depth')
    parser.add_argument('--edge-quantile',type=float,default=.93)
    parser.add_argument('--surface-band-m',type=float,default=.008)
    parser.add_argument('--depth-tolerance-m',type=float,default=.015)
    parser.add_argument('--max-reprojection-px',type=float,default=1.0)
    parser.add_argument('--lifting-step-px',type=int,default=2)
    parser.add_argument('--max-lifting-angle-deg',type=float,default=30.0)
    parser.add_argument('--min-overlap-voxels',type=int,default=10)
    args=parser.parse_args()
    if not args.views or args.views[0]!='front' or len(args.views)!=len(set(args.views)):
        parser.error('views must start with front and contain unique names')
    expected=(args.data_dir/'pde_governance/volume_partition_integration').resolve()
    if expected not in args.output_dir.resolve().parents:
        parser.error('output-dir must be under image pde_governance/volume_partition_integration/<run_id>')
    print(json.dumps(build_bundle(args),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
