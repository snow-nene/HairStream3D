"""用可见非头发面部拟合 FLAME 形状，独立输出头模及检查图，不修改 PDE。"""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.recon_3d.recon3D import load_calib


def fit_flame_head_experiment():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-id', default='0d285f5be7fa09c3dbbf1c9334047888')
    args = parser.parse_args()
    base = Path('results/multiview_data') / args.image_id
    out = base / 'pde_governance/flame_head_fit/step_01_face_shape_fit'
    out.mkdir(parents=True, exist_ok=True)
    with open('assets/FLAME/FLAME_NEUTRAL.pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')
    template = np.asarray(model['v_template'], dtype=float)
    basis = np.asarray(model['shapedirs'].r, dtype=float)[:, :, :50]
    face_ids = np.load('assets/FLAME/face_flame_index.npy').astype(int)
    head = o3d.io.read_triangle_mesh('data/head_model.obj')
    mesh = o3d.io.read_triangle_mesh(str(base / 'pixal3d/hair_mesh_aligned_best.obj'))
    raw = cv2.imread(str(base / 'raw_img.png'))
    seg = cv2.imread(str(base / 'maps/seg/front.png'), 0) > 127
    h, w = seg.shape
    calib = load_calib(str(base / 'maps/param/front_dense_silhouette.npy')).numpy().astype(float)
    inverse = np.linalg.inv(calib)
    # Initial pose/scale only: align the neutral template to the existing standard head.
    scale = np.ptp(np.asarray(head.vertices), axis=0)[0]/np.ptp(template, axis=0)[0]
    trans = head.get_axis_aligned_bounding_box().get_center()-scale*(template.min(0)+template.max(0))/2
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template*scale+trans))
    target = o3d.geometry.PointCloud(head.vertices)
    reg = o3d.pipelines.registration.registration_icp(
        source, target, .05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    rotation = reg.transformation[:3, :3]
    trans = rotation@trans+reg.transformation[:3, 3]
    initial = scale*template@rotation.T+trans
    initial_rotation, initial_scale, initial_trans = rotation.copy(), float(scale), trans.copy()
    clip = np.column_stack([initial, np.ones(len(initial))])@calib.T
    uv = (clip[:, :2]/clip[:, 3:4]+1)/2*[w-1, h-1]
    roi = np.zeros(seg.shape, np.uint8)
    cv2.fillConvexPoly(roi, cv2.convexHull(np.rint(uv[face_ids]).astype(np.int32)), 1)
    # Hair-adjacent pixels are excluded, not interpreted as bare scalp evidence.
    safe_mask = (roi > 0) & (distance_transform_edt(~seg) > 8)
    yy, xx = np.where(safe_mask)
    ndc = np.column_stack([xx/(w-1)*2-1, yy/(h-1)*2-1])
    mesh_clip = np.column_stack([np.asarray(mesh.vertices), np.ones(len(mesh.vertices))])@calib.T
    near_z, far_z = mesh_clip[:, 2].max()+1, mesh_clip[:, 2].min()-1
    near = np.column_stack([ndc, np.full(len(ndc), near_z), np.ones(len(ndc))])@inverse.T
    far = np.column_stack([ndc, np.full(len(ndc), far_z), np.ones(len(ndc))])@inverse.T
    near, far = near[:, :3]/near[:, 3:4], far[:, :3]/far[:, 3:4]
    direction = far-near
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hits = scene.cast_rays(o3d.core.Tensor(np.column_stack([near, direction]).astype(np.float32)))
    distance = hits['t_hit'].numpy()
    incidence = np.abs((hits['primitive_normals'].numpy()*direction).sum(1))
    hit = np.isfinite(distance) & (incidence > .35)
    observed = near[hit]+direction[hit]*distance[hit, None]
    tree = cKDTree(observed)
    # Freeze the eligible source vertices before fitting; no moving-mask score inflation.
    xy = np.rint(uv[face_ids]).astype(int)
    inside = ((xy >= 0) & (xy < [w, h])).all(1)
    eligible = face_ids[inside]
    xy = xy[inside]
    eligible = eligible[safe_mask[xy[:, 1], xy[:, 0]]]
    before, _ = tree.query(initial[eligible])
    eligible = eligible[before < .025]
    train = eligible[np.arange(len(eligible)) % 5 != 0]
    held = eligible[np.arange(len(eligible)) % 5 == 0]
    if len(train) < 100 or len(held) < 20:
        raise RuntimeError(f'可靠面部点不足: train={len(train)}, held={len(held)}')
    beta = np.zeros(50)
    history = []
    # Alternating robust correspondence, similarity pose and regularized shape fitting.
    # Shape prior: a 1-sigma coefficient incurs a 3mm residual penalty.
    for iteration in range(30):
        local = template+np.einsum('vck,k->vc', basis, beta)
        current = scale*local@rotation.T+trans
        distances, indices = tree.query(current[train])
        weights = np.minimum(1., .005/np.maximum(distances, 1e-9))
        good = distances < .025
        source_local = local[train[good]]
        dest = observed[indices[good]]
        weights = weights[good]
        weights /= weights.sum()
        a, b = (source_local*weights[:, None]).sum(0), (dest*weights[:, None]).sum(0)
        ac, bc = source_local-a, dest-b
        u, singular, vt = np.linalg.svd((bc*weights[:, None]).T@ac)
        sign = np.ones(3)
        sign[-1] = np.linalg.det(u@vt)
        rotation = u@np.diag(sign)@vt
        scale = float(np.clip((singular*sign).sum()/(weights[:, None]*ac**2).sum(),
                              initial_scale*.95, initial_scale*1.05))
        trans = b-scale*rotation@a
        transformed_basis = scale*np.einsum('ab,vbk->vak', rotation, basis[train[good]])
        residual = dest-(scale*template[train[good]]@rotation.T+trans)
        row_weight = np.repeat(np.sqrt(weights*len(weights)), 3)
        matrix = transformed_basis.reshape(-1, 50)*row_weight[:, None]
        rhs = residual.ravel()*row_weight
        beta = np.linalg.solve(matrix.T@matrix+(.003**2)*np.eye(50), matrix.T@rhs)
        beta = np.clip(beta, -2., 2.)
        history.append({'iteration': iteration, 'train_median_mm': float(np.median(distances)*1000),
                        'beta_norm': float(np.linalg.norm(beta))})
    fitted = scale*(template+np.einsum('vck,k->vc', basis, beta))@rotation.T+trans
    woeyes = np.load('assets/FLAME/woeyes_flame_index.npy').astype(int)
    closed_faces = np.load('assets/FLAME/closed_woeyes_flame_faces.npy').astype(int)
    outputs = {}
    for name, verts in [('initial_head', initial), ('fitted_head', fitted)]:
        result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts[woeyes]),
                                         o3d.utility.Vector3iVector(closed_faces))
        result.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out / f'{name}.obj'), result)
        outputs[name] = {'vertices': len(result.vertices), 'faces': len(result.triangles),
                         'watertight': result.is_watertight(),
                         'closed_edge_manifold': result.is_edge_manifold(allow_boundary_edges=False),
                         'self_intersection_pairs': len(result.get_self_intersecting_triangles())}
    np.savez_compressed(out / 'fit_parameters.npz', beta=beta, scale=scale, rotation=rotation,
                        translation=trans, initial_rotation=initial_rotation,
                        initial_scale=initial_scale, initial_translation=initial_trans,
                        train_vertex_ids=train, heldout_vertex_ids=held, observed_points=observed)
    report = {'image_id': args.image_id, 'shape_components': 50, 'expression_and_jaw': 'neutral/fixed',
              'target_points': len(observed), 'train_vertices': len(train), 'heldout_vertices': len(held),
              'beta_norm': float(np.linalg.norm(beta)),
              'beta_at_bound_count': int((np.abs(beta) >= 1.9999).sum()),
              'volume_subtraction_ready': False,
              'meshes': outputs, 'history': history,
              'note': '仅正面可见非头发区域拟合。留出的是源顶点而非独立真实头模；误差衡量对重建表面的贴合，不证明隐藏头型正确。未提取头发体积，未接入PDE。'}
    for name, verts in [('initial', initial), ('fitted', fitted)]:
        report[name] = {}
        for split, ids in [('train', train), ('heldout', held)]:
            distances, _ = tree.query(verts[ids])
            report[name][split] = dict(zip(['median_mm', 'p90_mm'], (np.quantile(distances, [.5, .9])*1000).tolist()))
    # Raw-image overlays: visible projected wireframe is a registration diagnostic only.
    panels = [raw.copy()]
    evidence = raw.copy()
    evidence[yy[hit], xx[hit]] = (.45*evidence[yy[hit], xx[hit]]+.55*np.array([70, 230, 70])).astype(np.uint8)
    panels.append(evidence)
    for verts in [initial, fitted]:
        overlay = raw.copy()
        clip = np.column_stack([verts, np.ones(len(verts))])@calib.T
        xy = np.rint((clip[:, :2]/clip[:, 3:4]+1)/2*[w-1, h-1]).astype(int)
        for point in xy[face_ids]:
            cv2.circle(overlay, tuple(point), 1, (20, 190, 255), -1)
        panels.append(overlay)
    for panel, title in zip(panels, ['Raw image', 'Green: fitting evidence', 'Initial FLAME face', 'Fitted FLAME face']):
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / 'front_comparison.png'), np.concatenate(panels, axis=1))
    # Fixed-axis views and thin cross-sections reveal invisible-head uncertainty.
    external = np.asarray(mesh.vertices)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (x, y), title in zip(axes[0], [(0, 1), (2, 1), (0, 2)], ['Front geometry', 'Side geometry', 'Top geometry']):
        ax.scatter(external[::30, x], external[::30, y], s=.15, c='gray', alpha=.3)
        ax.scatter(initial[::3, x], initial[::3, y], s=.3, c='royalblue', alpha=.5)
        ax.scatter(fitted[::3, x], fitted[::3, y], s=.3, c='darkorange', alpha=.6)
        ax.set_title(title+' (blue initial / orange fitted)')
        ax.set_aspect('equal')
    for ax, fraction in zip(axes[1], [.45, .65, .82]):
        level = fitted[:, 1].min()+fraction*np.ptp(fitted[:, 1])
        for verts, color, label in [(external, 'gray', 'Pixal3D'), (initial, 'royalblue', 'initial'), (fitted, 'darkorange', 'fitted')]:
            points = verts[np.abs(verts[:, 1]-level) < .003]
            ax.scatter(points[:, 0], points[:, 2], s=2, c=color, label=label)
        ax.set_title(f'Horizontal slice y={level:.3f}m (+/-3mm)')
        ax.set_aspect('equal')
        ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out / 'geometry_and_sections.png', dpi=150)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    fit_flame_head_experiment()
