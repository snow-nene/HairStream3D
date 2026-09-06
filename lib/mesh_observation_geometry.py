"""Calibrated mesh observation geometry; no neural-depth geometry or training."""
import numpy as np


def audit_orthographic_camera(camera):
    camera=np.asarray(camera,float)
    reasons=[]
    if camera.shape!=(4,4) or not np.isfinite(camera).all():
        return {'passed':False,'reasons':['invalid_matrix']}
    if not np.allclose(camera[3],[0,0,0,1],atol=1e-8):
        reasons.append('unsupported_perspective')
    scales=np.linalg.norm(camera[:3,:3],axis=1)
    if np.any(scales<1e-10):
        return {'passed':False,'reasons':reasons+['singular_camera']}
    axes=camera[:3,:3]/scales[:,None]
    orthogonality=float(np.linalg.norm(axes@axes.T-np.eye(3)))
    # HairStep's image y axis is down, while world coordinates are right handed.
    handedness=float(np.linalg.det(np.diag([1,-1,1])@axes))
    if orthogonality>1e-4:reasons.append('nonorthogonal_axes')
    if abs(handedness-1)>1e-4:reasons.append('axis_or_handedness_mismatch')
    return {'passed':not reasons,'reasons':reasons,'axis_scales':scales.tolist(),
            'orthogonality_error':orthogonality,'world_rotation_determinant':handedness}


def audit_registration(camera, image_shape, correspondences=None, max_q90_px=8., min_points=12):
    """Use independent correspondences, never a self-projection as validation.

    Correspondences must declare world metres, disjoint observation IDs and
    fit IDs, and a heldout boolean mask. This validates the supplied evidence,
    not its scientific provenance; callers must also bind its file hash.
    """
    report={'camera':audit_orthographic_camera(camera),'passed':False,
            'threshold_q90_px':float(max_q90_px),'minimum_points':min_points}
    if not report['camera']['passed']:
        report['reason']='invalid_camera';return report
    if correspondences is None:
        report['reason']='missing_independent_correspondences';return report
    if str(np.asarray(correspondences['units']).item())!='m':
        report['reason']='world_units_not_metres';return report
    world=np.asarray(correspondences['world'],float)
    pixels=np.asarray(correspondences['pixels'],float)
    heldout=np.asarray(correspondences['heldout'])
    ids=np.asarray(correspondences['observation_ids'])
    fit_ids=np.asarray(correspondences['fit_ids'])
    if (world.ndim!=2 or world.shape[1]!=3 or pixels.shape!=(len(world),2)
            or heldout.shape!=(len(world),) or heldout.dtype!=bool or ids.shape!=(len(world),)
            or not np.isfinite(world).all() or not np.isfinite(pixels).all()
            or len(np.unique(ids))!=len(ids)):
        report['reason']='invalid_correspondence_contract';return report
    if np.intersect1d(ids[heldout],fit_ids).size:
        report['reason']='holdout_used_in_fit';return report
    if heldout.sum()<min_points:
        report['reason']='insufficient_holdout';return report
    clip=np.c_[world,np.ones(len(world))]@np.asarray(camera).T
    projected=(clip[:,:2]/clip[:,3:]+1)*np.array([image_shape[1]-1,image_shape[0]-1])/2
    errors=np.linalg.norm(projected[heldout]-pixels[heldout],axis=1)
    report.update({'holdout_points':int(heldout.sum()),'q90_error_px':float(np.quantile(errors,.9)),
                   'median_error_px':float(np.median(errors))})
    report['passed']=bool(report['q90_error_px']<=max_q90_px)
    report['reason']='passed' if report['passed'] else 'holdout_reprojection_failed'
    return report


class VisibleHairRaycaster:
    """First hair intersections vetoed by the first head intersection."""
    def __init__(self,hair_mesh,head_mesh,camera,image_shape,min_incidence=.4):
        import open3d as o3d
        audit=audit_orthographic_camera(camera)
        if not audit['passed']:raise ValueError(str(audit))
        self.camera=np.asarray(camera,float);self.inverse=np.linalg.inv(self.camera)
        self.shape=tuple(image_shape);self.min_incidence=min_incidence
        self.hair=o3d.t.geometry.RaycastingScene();self.head=o3d.t.geometry.RaycastingScene()
        near=[]
        for path,scene in [(hair_mesh,self.hair),(head_mesh,self.head)]:
            mesh=o3d.io.read_triangle_mesh(str(path))
            vertices=np.asarray(mesh.vertices)
            if not len(vertices) or not len(mesh.triangles):raise ValueError(f'empty mesh: {path}')
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            near.append((np.c_[vertices,np.ones(len(vertices))]@self.camera.T)[:,2].max())
        self.near_z=max(near)+1

    def cast_pixels(self,pixels):
        import open3d as o3d
        pixels=np.asarray(pixels,float)
        if pixels.ndim!=2 or pixels.shape[1]!=2 or not np.isfinite(pixels).all():
            raise ValueError('pixels must be finite N-by-2')
        uv=2*pixels/np.array([self.shape[1]-1,self.shape[0]-1])-1
        origins=np.c_[uv,np.full(len(uv),self.near_z),np.ones(len(uv))]@self.inverse.T
        origins=origins[:,:3]/origins[:,3:]
        ray=-self.inverse[:3,2];ray/=np.linalg.norm(ray)
        rays=o3d.core.Tensor(np.c_[origins,np.tile(ray,(len(origins),1))].astype(np.float32))
        hair=self.hair.cast_rays(rays);head=self.head.cast_rays(rays)
        distance=hair['t_hit'].numpy();head_distance=head['t_hit'].numpy()
        normals=hair['primitive_normals'].numpy()
        hit=np.isfinite(distance)
        occluded=hit&(head_distance<=distance)
        grazing=hit&(np.abs(normals@ray)<self.min_incidence)
        in_image=((pixels>=0)&(pixels<=np.array([self.shape[1]-1,self.shape[0]-1]))).all(1)
        valid=hit&~occluded&~grazing&in_image
        world=np.full((len(pixels),3),np.nan)
        world[hit]=origins[hit]+distance[hit,None]*ray
        return {'world':world,'normal':normals,'face':hair['primitive_ids'].numpy(),
                'valid':valid,'hair_hit':hit,'head_occluded':occluded,'grazing':grazing,
                'outside_image':~in_image}


def differential_visible_lift(raycaster,pixels,directions,labels,edge,
                              delta_px=2.,max_distance_m=.008,max_angle_deg=20.):
    """Two actual surface points with image-path partition and edge guards."""
    pixels=np.asarray(pixels,float);directions=np.asarray(directions,float)
    if pixels.shape!=directions.shape or pixels.ndim!=2 or pixels.shape[1]!=2:
        raise ValueError('pixels and directions must be N-by-2')
    if delta_px<=0 or max_distance_m<=0 or not 0<max_angle_deg<90:
        raise ValueError('invalid lifting thresholds')
    norm=np.linalg.norm(directions,axis=1)
    direction=directions/np.maximum(norm[:,None],1e-12)
    first=raycaster.cast_pixels(pixels)
    following=raycaster.cast_pixels(pixels+delta_px*direction)
    vector=following['world']-first['world']
    distance=np.linalg.norm(vector,axis=1)
    tangent=vector/np.maximum(distance[:,None],1e-12)
    # Analytic lift preserves image direction and lies in the local tangent plane.
    pixel_scale=2/np.array([raycaster.shape[1]-1,raycaster.shape[0]-1])
    base=np.c_[direction*pixel_scale,np.zeros(len(direction))]@raycaster.inverse[:3,:3].T
    ray=raycaster.inverse[:3,2]
    denom=first['normal']@ray
    factor=np.divide(-np.sum(first['normal']*base,axis=1),denom,
                     out=np.zeros(len(base)),where=np.abs(denom)>1e-10)
    analytic=base+factor[:,None]*ray
    analytic/=np.maximum(np.linalg.norm(analytic,axis=1,keepdims=True),1e-12)
    dot=np.clip(np.abs(np.sum(analytic*tangent,axis=1)),0,1)
    angle=np.degrees(np.arccos(dot))
    shape=np.array([labels.shape[1],labels.shape[0]])
    start=np.clip(np.rint(pixels).astype(int),0,shape-1)
    identity=labels[start[:,1],start[:,0]]
    path_valid=identity>0
    path_geometry_valid=first["valid"].copy()
    previous_world=first["world"]
    path_length=np.zeros(len(pixels))
    for t in np.linspace(0,1,int(np.ceil(delta_px*2))+1):
        sample_pixels=pixels+t*delta_px*direction
        sample=np.rint(sample_pixels).astype(int)
        surface=raycaster.cast_pixels(sample_pixels)
        path_geometry_valid &= surface['valid']
        path_length += np.linalg.norm(surface['world']-previous_world,axis=1)
        previous_world=surface['world']
        inside=((sample>=0)&(sample<shape)).all(1)
        sample=np.clip(sample,0,shape-1)
        path_valid &= inside&(labels[sample[:,1],sample[:,0]]==identity)&~edge[sample[:,1],sample[:,0]]
    reasons={'visibility_rejected':~first['valid']|~following['valid']|~path_geometry_valid,
             'partition_or_edge_rejected':~path_valid,
             'geometry_jump_or_zero':~np.isfinite(distance)|(distance>max_distance_m)|(distance<1e-8)|~np.isfinite(path_length)|(path_length>max_distance_m),
             'tangent_disagreement':~np.isfinite(angle)|(angle>max_angle_deg)|(np.abs(denom)<1e-10),
             'invalid_direction':~np.isfinite(norm)|(norm<1e-8)}
    accepted=~np.logical_or.reduce(list(reasons.values()))
    return {'world':first['world'],'direction':tangent,'analytic_direction':analytic,
            'partition_labels':identity,'accepted':accepted,'distance_m':distance,
            'angle_degrees':angle,**reasons}
