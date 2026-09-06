"""定位固定留出集上的网格/Depth Pro 深度不一致，不修改融合输入。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates, distance_transform_edt, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib


def audit_depth_fit_hotspots():
    base = Path("results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888")
    root = base / "pde_governance/sparse_guides"
    out = root / "step_07_depth_hotspots"
    out.mkdir(parents=True, exist_ok=True)
    fit = json.loads((root / "step_06_depth_fusion/report.json").read_text())
    guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
    evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
    depth = np.load(root / "step_05_matted_depth_pro/corrected_hair_depth.npy")
    camera = load_calib(str(base / "maps/param/front_dense_silhouette.npy")).numpy().astype(float)
    raw = cv2.imread(str(base / "raw_img.png"))
    seg = cv2.imread(str(base / "maps/seg/front.png"), 0) > 127
    dist = distance_transform_edt(seg)
    labels = np.load(base / "pde_governance/strand_depth_partition/step_01_edge_partition_audit/hybrid_prototype_labels.npy")
    arrays = []
    curve_ids = []
    for key in fit["validation_guides"]:
        xy = guides[key][:, :2]
        world = evidence[key + "_world_raw"]
        predicted = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        border = map_coordinates(dist, [xy[:, 1], xy[:, 0]], order=1)
        incidence = evidence[key + "_incidence"]
        jump = np.linalg.norm(np.diff(world, axis=0), axis=1) > .01
        suspect = np.zeros(len(world), bool)
        suspect[:-1] |= jump; suspect[1:] |= jump
        reliable = evidence[key + "_hit_mask"] & (incidence > .3) & (border > 5) & ~binary_dilation(suspect, iterations=2) & np.isfinite(predicted)
        z = (np.column_stack([world, np.ones(len(world))]) @ camera.T)[:, 2]
        residual = (predicted * fit["affine_scale_offset"][0] + fit["affine_scale_offset"][1] - z) * fit["meters_per_camera_z"] * 1000
        partition = map_coordinates(labels, [xy[:, 1], xy[:, 0]], order=0)
        arrays.append(np.column_stack([xy, world, residual, incidence, border, partition])[reliable])
        curve_ids.extend([key] * int(reliable.sum()))
    values = np.concatenate(arrays)
    assert len(values) == fit["validation_points"]
    absolute = np.abs(values[:, 5])
    head = o3d.io.read_triangle_mesh("data/head_model.obj")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head))
    head_distance = scene.compute_signed_distance(o3d.core.Tensor(values[:, 2:5].astype(np.float32))).numpy()
    groups = {"all": np.ones(len(values), bool), "high_gt20mm": absolute > 20,
              "extreme_gt50mm": absolute > 50, "low_le20mm": absolute <= 20}
    for label in np.unique(values[:, 8]):
        groups[f"partition_{int(label)}"] = values[:, 8] == label
    y0, y1 = np.where(seg)[0].min(), np.where(seg)[0].max()
    for index, name in enumerate(("upper", "middle", "lower")):
        groups[name] = (values[:, 1] >= y0 + (y1-y0+1)*index/3) & (values[:, 1] < y0+(y1-y0+1)*(index+1)/3)
    report = {"groups": {}, "scope": "与Step6完全相同的2260个留出点；高误差为>20mm。头模负有向距离仅作线索，不等于语义误命中。"}
    for name, selected in groups.items():
        if not selected.any():
            continue
        report["groups"][name] = {
            "count": int(selected.sum()), "abs_error_mm_q50_q90": np.quantile(absolute[selected], [.5,.9]).tolist(),
            "signed_error_mm_median": float(np.median(values[selected, 5])),
            "high_fraction": float(np.mean(absolute[selected] > 20)),
            "incidence_median": float(np.median(values[selected, 6])),
            "distance_to_seg_border_px_median": float(np.median(values[selected, 7])),
            "inside_template_head_gt2mm": int((head_distance[selected] < -.002).sum())}
    overlay = raw.copy()
    signed_overlay = raw.copy()
    for i in np.argsort(absolute):
        point = tuple(np.rint(values[i, :2] * [raw.shape[1]/seg.shape[1],raw.shape[0]/seg.shape[0]]).astype(int))
        color = (70,210,70) if absolute[i] <= 20 else ((0,180,255) if absolute[i] <= 50 else (30,30,255))
        cv2.circle(overlay, point, 2, color, -1)
        if absolute[i] > 20:
            cv2.circle(signed_overlay, point, 2, (30,30,255) if values[i,5]>0 else (255,160,30), -1)
    cv2.putText(overlay,"Green <=20 / Orange 20-50 / Red >50 mm",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(signed_overlay,"Red: fit nearer / Blue: fit farther",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out / "hotspots_on_raw.png"), np.concatenate([raw,overlay,signed_overlay],axis=1))
    worst = np.argsort(absolute)[-20:][::-1]
    report["worst_points"] = [{"guide":curve_ids[i],"xy":values[i,:2].tolist(),"signed_error_mm":float(values[i,5]),"template_signed_distance_mm":float(head_distance[i]*1000)} for i in worst]
    np.savez_compressed(out / "validation_points.npz", xy=values[:,:2], world=values[:,2:5], residual_mm=values[:,5], incidence=values[:,6], border_px=values[:,7], partition=values[:,8], head_signed_distance_m=head_distance)
    (out / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report["groups"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    audit_depth_fit_hotspots()
