"""固定曲线留出集，比较全局与左右分区仿射深度校准。"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib


def compare_partition_depth_calibration():
    base = Path("results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888")
    root = base / "pde_governance/sparse_guides"
    out = root / "step_08_partition_calibration"
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root / "step_06_depth_fusion/report.json").read_text())
    guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
    evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
    depth = np.load(root / "step_05_matted_depth_pro/corrected_hair_depth.npy")
    labels = np.load(base / "pde_governance/strand_depth_partition/step_01_edge_partition_audit/hybrid_prototype_labels.npy")
    raw = cv2.imread(str(base / "raw_img.png"))
    seg = cv2.imread(str(base / "maps/seg/front.png"), 0) > 127
    border_distance = distance_transform_edt(seg)
    camera = load_calib(str(base / "maps/param/front_dense_silhouette.npy")).numpy().astype(float)
    rows = []
    for index, key in enumerate(guides.files):
        xy = guides[key][:, :2]
        world = evidence[key + "_world_raw"]
        predicted = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        border = map_coordinates(border_distance, [xy[:, 1], xy[:, 0]], order=1)
        jump = np.linalg.norm(np.diff(world, axis=0), axis=1) > .01
        suspect = np.zeros(len(world), bool)
        suspect[:-1] |= jump
        suspect[1:] |= jump
        valid = evidence[key + "_hit_mask"] & (evidence[key + "_incidence"] > .3) & (border > 5) & ~binary_dilation(suspect, iterations=2) & np.isfinite(predicted)
        target = (np.column_stack([world, np.ones(len(world))]) @ camera.T)[:, 2]
        partition = map_coordinates(labels, [xy[:, 1], xy[:, 0]], order=0)
        rows.append(np.column_stack([xy, predicted, target, partition, np.full(len(xy), index),
                                     np.full(len(xy), key in previous["validation_guides"])])[valid])
    data = np.concatenate(rows)
    validation = data[:, 6].astype(bool)
    assert validation.sum() == previous["validation_points"]
    assert (~validation).sum() == previous["train_points"]
    coefficients = {"0": previous["affine_scale_offset"]}
    models = {}
    for label in (1, 2):
        train = ~validation & (data[:, 4] == label)
        assert train.sum() > 30
        design = np.column_stack([data[train, 2], np.ones(train.sum())])
        target = data[train, 3]
        weights = np.ones(train.sum())
        for _ in range(10):
            coefficient = np.linalg.lstsq(design * np.sqrt(weights[:, None]), target * np.sqrt(weights), rcond=None)[0]
            residual = design @ coefficient - target
            sigma = max(1.4826*np.median(np.abs(residual-np.median(residual))),1e-5)
            weights = np.minimum(1,1.345*sigma/np.maximum(np.abs(residual),1e-10))
        coefficients[str(label)] = coefficient.tolist()
        models[str(label)] = {"train_points": int(train.sum()), "train_guides": int(len(np.unique(data[train,5]))),
                              "scale_offset": coefficient.tolist(), "expected_negative_slope": bool(coefficient[0]<0)}
    factor = previous["meters_per_camera_z"] * 1000
    global_prediction = data[:, 2]*previous["affine_scale_offset"][0]+previous["affine_scale_offset"][1]
    local_prediction = global_prediction.copy()
    for label in (1,2):
        chosen = data[:,4] == label
        a,b = coefficients[str(label)]
        local_prediction[chosen] = data[chosen,2]*a+b
    global_error = np.abs(global_prediction-data[:,3])*factor
    local_error = np.abs(local_prediction-data[:,3])*factor
    # Step6 的采样深度保留 float32，此处汇总为 float64；允许0.0001mm舍入误差。
    np.testing.assert_allclose(np.quantile(global_error[validation],[.5,.9]),previous["validation_error_mm_q50_q90"],atol=1e-4)
    report = {"models":models,"groups":{},"notes":"相同留出曲线复用以比较假设，已非未见最终测试集；所有拟合只使用训练曲线。label0保持全局回退。网格不是深度真值。"}
    for name, group in (("all",np.ones(len(data),bool)),("partition_1",data[:,4]==1),("partition_2",data[:,4]==2),("unassigned",data[:,4]==0)):
        selected = validation & group
        if not selected.any():
            continue
        report["groups"][name] = {"validation_points":int(selected.sum()),
            "global_error_mm_q50_q90":np.quantile(global_error[selected],[.5,.9]).tolist(),
            "partition_error_mm_q50_q90":np.quantile(local_error[selected],[.5,.9]).tolist(),
            "global_gt20mm_fraction":float(np.mean(global_error[selected]>20)),
            "partition_gt20mm_fraction":float(np.mean(local_error[selected]>20))}
    report["passed_20mm_p90"] = bool(np.quantile(local_error[validation],.9)<=20 and all(m["expected_negative_slope"] for m in models.values()))
    report["validation_points_worsened_gt5mm"] = int(np.sum(validation & (local_error-global_error>5)))
    images = [raw]
    for error, title in ((global_error,"Global calibration"),(local_error,"Partition calibration")):
        image = raw.copy()
        indices = np.flatnonzero(validation)
        for i in indices[np.argsort(error[indices])]:
            xy = np.rint(data[i,:2]*[raw.shape[1]/seg.shape[1],raw.shape[0]/seg.shape[0]]).astype(int)
            color = (70,210,70) if error[i]<=20 else ((0,180,255) if error[i]<=50 else (30,30,255))
            cv2.circle(image,tuple(xy),2,color,-1)
        cv2.putText(image,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        images.append(image)
    cv2.imwrite(str(out / "comparison_on_raw.png"),np.concatenate(images,axis=1))
    np.savez_compressed(out / "calibration_evidence.npz",data=data,global_error_mm=global_error,partition_error_mm=local_error)
    (out / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    compare_partition_depth_calibration()
