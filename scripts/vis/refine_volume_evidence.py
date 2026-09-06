"""像素表面交点体素化与孤立空气证据复核，不填充隐藏头发。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, convolve, label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import trimesh
from matplotlib.collections import LineCollection
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration


def refine_volume_evidence():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    parent = base / 'pde_governance/volume_domain_audit'
    old = np.load(parent / 'step_06_four_class_evidence/volume_evidence.npz')
    out = parent / 'step_07_surface_evidence_continuity'
    out.mkdir(parents=True, exist_ok=True)
    original = old['labels']
    low, high = old['b_min'], old['b_max']
    shape = np.array(original.shape)
    spacing = (high-low)/(shape-1)
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    votes = np.zeros(shape, np.uint8)
    weights = []
    stats = {}
    for view in ['front', 'left', 'right', 'back']:
        calib = (load_calib(str(base / 'maps/param/front_dense_silhouette.npy')).numpy() if view == 'front'
                 else load_blender_view_calibration(str(base), view)[0]).astype(float)
        mask = cv2.imread(str(base / 'maps/seg' / f'{view}.png'), 0)>127
        py, px = np.where(distance_transform_edt(mask)>=3)
        uv = np.column_stack([px/(mask.shape[1]-1)*2-1, py/(mask.shape[0]-1)*2-1])
        clip = np.column_stack([np.asarray(mesh.vertices), np.ones(len(mesh.vertices))])@calib.T
        near_depth = clip[:, 2].max()+1
        inverse = np.linalg.inv(calib)
        near = np.column_stack([uv, np.full(len(uv), near_depth), np.ones(len(uv))])@inverse.T
        far = np.column_stack([uv, np.full(len(uv), near_depth-1), np.ones(len(uv))])@inverse.T
        near, far = near[:, :3]/near[:, 3:4], far[:, :3]/far[:, 3:4]
        direction = far-near
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        result = scene.cast_rays(o3d.core.Tensor(np.column_stack([near, direction]).astype(np.float32)))
        depth = result['t_hit'].numpy()
        incidence = np.abs((result['primitive_normals'].numpy()*direction).sum(1))
        valid = np.isfinite(depth) & (incidence>.4)
        hits = near[valid]+direction[valid]*depth[valid, None]
        grid = (hits-low)/spacing
        floor = np.floor(grid).astype(int)
        fraction = grid-floor
        support = np.zeros(shape, np.float32)
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    offset = np.array([x, y, z])
                    ids = floor+offset
                    weight = np.prod(np.where(offset, fraction, 1-fraction), axis=1)
                    inside = ((ids>=0)&(ids<shape)).all(1)
                    np.maximum.at(support, tuple(ids[inside].T), weight[inside])
        # Max contribution per view prevents pixel density from masquerading as multiview evidence.
        votes += support>=.08
        weights.append(support)
        stats[view] = {'valid_surface_hits': int(valid.sum()), 'supported_voxels': int((support>=.08).sum())}
    hair = (weights[0]>=.08) | (votes>=2) | (original==1)
    axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    wrap = o3d.io.read_triangle_mesh(str(parent / 'step_04_root_ray_consistency/full_model_outer_wrap_large.off'))
    wrap_scene = o3d.t.geometry.RaycastingScene()
    wrap_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wrap))
    outside = wrap_scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=5).numpy().reshape(shape)>.006
    air_before = original==3
    neighbor = convolve(air_before.astype(np.int16), np.ones((3,3,3), np.int16), mode='constant', cval=0)
    isolated_air = air_before & ~outside & (neighbor<18)
    air = air_before & ~isolated_air
    head = original==2
    conflict = old['conflict'] | (hair & (air | head))
    labels = np.zeros(shape, np.uint8)
    labels[hair & ~conflict] = 1
    labels[head & ~conflict] = 2
    labels[air & ~conflict] = 3
    assert (labels[conflict]==0).all()
    assert (labels[isolated_air & ~hair & ~head]==0).all()
    report = {'per_view':stats, 'air_returned_to_unknown_candidates':int(isolated_air.sum()),
              'conflict_voxels':int(conflict.sum()), 'before':{}, 'after':{},
              'note':'表面像素射线交点对8邻体素做三线性贡献，不是精确三角形-体素相交；最大权重>=0.08支持。仍保留front或两视图规则，单视图弱证据单独保存。内部空气需18/27邻域支持，否则退回未知；此为待验证启发式，不是真实空腔判定。未做全域分割或PDE。'}
    for name, array in [('before',original),('after',labels)]:
        _, components = label(array==1)
        report[name]={'class_counts':{str(i):int((array==i).sum()) for i in range(4)}, 'hair_components_6_connected':components}
    np.savez_compressed(out/'refined_evidence.npz', labels=labels, conflict=conflict,
                        per_view_surface_weights=np.stack(weights), weak_surface_evidence=(votes>0)&~hair,
                        b_min=low,b_max=high)
    tri = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    fig, axs = plt.subplots(2,3,figsize=(16,10))
    for row, data in enumerate([original,labels]):
        for ax,(axis,value) in zip(axs[row],[(1,1.78),(1,1.84),(0,0)]):
            index=np.argmin(np.abs(axes[axis]-value))
            hor,ver=(0,2) if axis==1 else (2,1)
            remaining=[i for i in range(3) if i!=axis]
            display=data.copy()
            display[old['conflict'] if row==0 else conflict]=4
            plane=np.take(display,index,axis=axis).transpose(remaining.index(ver),remaining.index(hor))
            ax.imshow(plane,origin='lower',extent=[low[hor]-spacing[hor]/2,high[hor]+spacing[hor]/2,low[ver]-spacing[ver]/2,high[ver]+spacing[ver]/2],
                cmap=ListedColormap(['#ffdc78','#64bf72','#80aee8','#f3f3f3','#c167ca']),vmin=0,vmax=4,interpolation='nearest')
            origin,normal=np.zeros(3),np.zeros(3)
            origin[axis],normal[axis]=axes[axis][index],1
            lines=trimesh.intersections.mesh_plane(tri,plane_normal=normal,plane_origin=origin)
            ax.add_collection(LineCollection(lines[:,:,[hor,ver]],colors='black',linewidths=1))
            ax.set_aspect('equal')
            ax.set_title(f'{"Before" if row==0 else "After"} | {"XYZ"[axis]}={origin[axis]:.3f}m')
    fig.suptitle('Green hair evidence | Blue head prior | Gray air | Yellow unknown | Purple conflict')
    fig.tight_layout(rect=[0,0,1,.97])
    fig.savefig(out/'before_after_sections.png',dpi=150)
    plt.close(fig)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    refine_volume_evidence()
