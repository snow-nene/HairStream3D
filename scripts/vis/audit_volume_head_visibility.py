"""用显式头模首交点区分可见发丝泄漏与头后投影；保留原投影指标。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib


def visible_raster_metrics(strands,camera,seg,head_path):
    height,width=seg.shape
    mesh=o3d.io.read_triangle_mesh(str(head_path))
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    inverse=np.linalg.inv(camera)
    y,x=np.mgrid[:height,:width]
    uv=np.c_[2*x.ravel()/(width-1)-1,2*y.ravel()/(height-1)-1]
    mesh_clip=np.c_[np.asarray(mesh.vertices),np.ones(len(mesh.vertices))]@camera.T
    near_z=float(mesh_clip[:,2].max()+1)
    near=np.c_[uv,np.full(len(uv),near_z),np.ones(len(uv))]@inverse.T
    far=np.c_[uv,np.full(len(uv),near_z-1),np.ones(len(uv))]@inverse.T
    near=near[:,:3]/near[:,3:];far=far[:,:3]/far[:,3:]
    ray=far-near;ray/=np.linalg.norm(ray,axis=1,keepdims=True)
    hits=scene.cast_rays(o3d.core.Tensor(np.c_[near,ray].astype(np.float32)))['t_hit'].numpy()
    valid=np.isfinite(hits)
    head_z=np.full(height*width,-np.inf)
    hit_world=near[valid]+ray[valid]*hits[valid,None]
    head_z[valid]=(np.c_[hit_world,np.ones(len(hit_world))]@camera.T)[:,2]
    # Orthographic camera: dense samples include both segment endpoints. Keep
    # the frontmost strand camera-z per pixel before comparing head occlusion.
    flat=strands.reshape(-1,3)
    clip=(np.c_[flat,np.ones(len(flat))]@camera.T).reshape(*strands.shape[:2],4)
    projected=clip[...,:3]/clip[...,3:]
    projected[...,:2]=(projected[...,:2]+1)*[.5*(width-1),.5*(height-1)]
    zbuffer=np.full(height*width,-np.inf)
    offscreen=set()
    for step in range(strands.shape[1]-1):
        a=projected[:,step];delta=projected[:,step+1]-a
        count=np.maximum(1,np.ceil(np.max(np.abs(delta[:,:2]),axis=1)*2).astype(int))
        for substep in range(int(count.max())+1):
            active=substep<=count
            point=a[active]+delta[active]*(substep/count[active])[:,None]
            pixel=np.rint(point[:,:2]).astype(int)
            inside=((pixel>=[0,0])&(pixel<[width,height])).all(1)
            np.maximum.at(zbuffer,pixel[inside,1]*width+pixel[inside,0],point[inside,2])
            offscreen.update(map(tuple,pixel[~inside].tolist()))
    drawn=np.isfinite(zbuffer)
    # This is an additional prior-based diagnostic, not a replacement metric.
    visible=drawn&(zbuffer>=head_z)
    hidden=drawn&~visible
    hair=seg.ravel()
    report={'raster':'segment samples at <=0.5 pixel, frontmost strand z per pixel',
        'head_prior':str(head_path),'head_sha256':hashlib.sha256(head_path.read_bytes()).hexdigest(),
        'visible_coverage':float((visible&hair).sum()/hair.sum()),
        'visible_leakage':float(((visible&~hair).sum()+len(offscreen))/(visible.sum()+len(offscreen))),
        'hidden_projected_pixels':int(hidden.sum()),'visible_pixels':int(visible.sum()),
        'offscreen_pixels':len(offscreen),'prior_not_ground_truth':True}
    image=np.zeros((height,width,3),np.uint8);image[seg]=35
    image.reshape(-1,3)[hidden]=[180,80,20]
    image.reshape(-1,3)[visible&hair]=[90,210,90]
    image.reshape(-1,3)[visible&~hair]=[50,50,255]
    return report,image


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--head-mesh',type=Path,default=Path('data/head_model.obj'))
    args=parser.parse_args()
    expected=(args.data_dir/'pde_governance/volume_partition_integration').resolve()
    if expected not in args.run_dir.resolve().parents:
        raise ValueError('run-dir must belong to the specified image')
    camera=load_calib(str(args.data_dir/'maps/param/front_dense_silhouette.npy')).numpy().astype(float)
    seg=cv2.imread(str(args.data_dir/'maps/seg/front.png'),0)>127
    strands=np.load(args.run_dir/'connected_strands.npz')['strands']
    report,overlay=visible_raster_metrics(strands,camera,seg,args.head_mesh)
    (args.run_dir/'head_visibility_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    cv2.imwrite(str(args.run_dir/'head_visibility_overlay.png'),overlay)
    print(json.dumps(report,ensure_ascii=False,indent=2))
