"""四视图可见性分割头发外表面；未观测面片近邻补标，不伪称封闭体积。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.recon_3d.recon3D import load_calib
from scripts.recon_3d.run_pde_multiview import load_blender_view_calibration


def extract_complete_hair_experiment():
    base = Path('results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888')
    out = base / 'pde_governance/hair_mesh_extraction/step_01_multiview_surface'
    out.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centers = vertices[faces].mean(1)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    views = ['front', 'left', 'right', 'back']
    calibs = {'front': load_calib(str(base / 'maps/param/front_dense_silhouette.npy')).numpy()}
    for view in views[1:]:
        calibs[view] = load_blender_view_calibration(str(base), view)[0]
    score = np.zeros(len(faces))
    support = np.zeros(len(faces))
    front_label = np.zeros(len(faces), np.int8)
    records = {}
    for view in views:
        mask = cv2.imread(str(base / 'maps/seg' / f'{view}.png'), 0) > 127
        signed = distance_transform_edt(mask)-distance_transform_edt(~mask)
        calib = calibs[view].astype(float)
        inverse = np.linalg.inv(calib)
        clip = np.column_stack([centers, np.ones(len(centers))])@calib.T
        ndc = clip[:, :2]/clip[:, 3:4]
        near_depth = float((np.column_stack([vertices, np.ones(len(vertices))])@calib.T)[:, 2].max()+1)
        near = np.column_stack([ndc, np.full(len(ndc), near_depth), np.ones(len(ndc))])@inverse.T
        near = near[:, :3]/near[:, 3:4]
        direction = centers-near
        distance = np.linalg.norm(direction, axis=1)
        direction /= distance[:, None]
        result = scene.cast_rays(o3d.core.Tensor(np.column_stack([near, direction]).astype(np.float32)))
        hit_depth = result['t_hit'].numpy()
        incidence = np.abs((result['primitive_normals'].numpy()*direction).sum(1))
        visible = (np.abs(hit_depth-distance) < .0015) & (np.abs(ndc) <= 1).all(1) & (incidence > .15)
        px = (ndc[:, 0]+1)/2*(mask.shape[1]-1)
        py = (ndc[:, 1]+1)/2*(mask.shape[0]-1)
        evidence = map_coordinates(signed, [py, px], order=1, mode='constant', cval=0)
        weight = visible*np.clip(np.abs(evidence)/4, 0, 1)*incidence*(3 if view == 'front' else 1)
        score += weight*np.sign(evidence)
        support += weight
        if view == 'front':
            reliable = visible & (np.abs(evidence) >= 3)
            front_label[reliable] = np.sign(evidence[reliable]).astype(np.int8)
        records[view] = {'visible_faces': int(visible.sum()), 'positive_faces': int((visible & (evidence > 0)).sum())}
        print(view, records[view], flush=True)
    observed = support > .05
    keep = score > 0
    keep[front_label != 0] = front_label[front_label != 0] > 0
    # Unseen labels are estimates, bounded to a local 15mm surface neighborhood.
    tree = cKDTree(centers[observed])
    missing = np.flatnonzero(~observed)
    dist, nearest = tree.query(centers[missing])
    keep[missing] = keep[observed][nearest] & (dist < .015)
    np.savez_compressed(out / 'face_classification.npz', selected=keep, observed=observed,
                        score=score, support=support, front_label=front_label,
                        unobserved_ids=missing, nearest_evidence_distance=dist)
    selected = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.copy()),
                                        o3d.utility.Vector3iVector(faces[keep].copy()))
    selected.remove_duplicated_vertices()
    selected.remove_degenerate_triangles()
    selected.remove_duplicated_triangles()
    selected.remove_unreferenced_vertices()
    selected.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / 'hair_surface_full.obj'), selected)
    hair_scene = o3d.t.geometry.RaycastingScene()
    hair_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(selected))
    panels = []
    reprojection = {}
    for view in views:
        size = 640
        xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        uv = np.column_stack([xx.ravel(), yy.ravel()])
        calib = calibs[view].astype(float)
        inverse = np.linalg.inv(calib)
        near_depth = float((np.column_stack([vertices, np.ones(len(vertices))])@calib.T)[:, 2].max()+1)
        near = np.column_stack([uv, np.full(len(uv), near_depth), np.ones(len(uv))])@inverse.T
        far = np.column_stack([uv, np.full(len(uv), near_depth-1), np.ones(len(uv))])@inverse.T
        near, far = near[:, :3]/near[:, 3:4], far[:, :3]/far[:, 3:4]
        direction = far-near
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        rays = o3d.core.Tensor(np.column_stack([near, direction]).astype(np.float32))
        original = scene.cast_rays(rays)
        result = hair_scene.cast_rays(rays)
        hit = np.isfinite(result['t_hit'].numpy())
        original_depth = original['t_hit'].numpy()
        visible = hit & (result['t_hit'].numpy() <= original_depth+.002)
        normals = result['primitive_normals'].numpy()
        shade = .35+.65*np.abs(normals@np.array([.3, .5, .8124]))
        canvas = np.full((size*size, 3), 242, np.uint8)
        canvas[hit] = np.clip(shade[hit, None]*[70, 170, 230], 0, 255).astype(np.uint8)
        canvas = canvas.reshape(size, size, 3)
        cv2.putText(canvas, f'{view}: extracted hair surface', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'{view}.png'), canvas)
        panels.append(canvas)
        target = cv2.resize(cv2.imread(str(base / 'maps/seg' / f'{view}.png'), 0), (size, size), interpolation=cv2.INTER_NEAREST) > 127
        prediction = visible.reshape(size, size)
        reprojection[view] = {'visible_mask_iou': float((prediction & target).sum()/max(1, (prediction | target).sum()))}
        if view == 'front':
            raw = cv2.resize(cv2.imread(str(base / 'raw_img.png')), (size, size))
            raw[prediction] = (.55*raw[prediction]+.45*canvas[prediction]).astype(np.uint8)
            cv2.imwrite(str(out / 'front_on_raw.png'), raw)
    cv2.imwrite(str(out / 'four_views.png'), np.concatenate(panels, axis=1))
    components, counts, areas = selected.cluster_connected_triangles()
    report = {'source_faces': len(faces), 'selected_source_faces': int(keep.sum()),
              'selected_unobserved_faces': int((keep & ~observed).sum()),
              'output_vertices': len(selected.vertices), 'output_faces': len(selected.triangles),
              'components': len(counts), 'largest_component_faces': int(np.max(counts)),
              'watertight': selected.is_watertight(),
              'boundary_or_nonmanifold_edges': len(selected.get_non_manifold_edges(allow_boundary_edges=False)),
              'per_view': records, 'reprojection': reprojection,
              'note': '四视图seg和遮挡分类原mesh面片，正面可靠标签优先；未观测区最多15mm近邻补标。侧背seg是生成视图证据，非真实独立观测。完整视角外表面尝试，不保证隐藏部分完整，不是封闭头发体积。未使用自相交FLAME做布尔差，未修改PDE。'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    extract_complete_hair_experiment()
