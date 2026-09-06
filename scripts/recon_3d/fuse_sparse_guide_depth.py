"""先留出曲线检验深度标定，通过后才沿相机射线融合可见曲线。"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, binary_dilation
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recon_3d.recon3D import load_calib
from lib.hair_util import save_polyline_strands


def run_sparse_depth_fusion():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--validation-p90-mm", type=float, default=20)
    args = parser.parse_args()
    base = Path("results/multiview_data") / args.image_id
    root = base / "pde_governance/sparse_guides"
    out = root / "step_06_depth_fusion"
    out.mkdir(parents=True, exist_ok=True)
    guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
    evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
    depth = np.load(root / "step_05_matted_depth_pro/corrected_hair_depth.npy")
    confidence = np.load(root / "step_05_matted_depth_pro/depth_confidence.npy")
    seg = cv2.imread(str(base / "maps/seg/front.png"), 0) > 127
    distance = distance_transform_edt(seg)
    camera = load_calib(str(base / "maps/param/front_dense_silhouette.npy")).numpy().astype(float)
    inverse = np.linalg.inv(camera)
    # 参数 z 的单位转换为沿该正交相机射线的物理米。
    ray_m_per_z = np.linalg.norm(inverse[:3, 2])
    records = []
    for index, key in enumerate(guides.files):
        xy = guides[key][:, :2]
        world = evidence[key + "_world_raw"]
        hit = evidence[key + "_hit_mask"]
        predicted = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        conf = map_coordinates(confidence, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        border = map_coordinates(distance, [xy[:, 1], xy[:, 0]], order=1)
        z = (np.column_stack([world, np.ones(len(world))]) @ camera.T)[:, 2]
        jump = np.linalg.norm(np.diff(world, axis=0), axis=1) > 0.01
        suspect = np.zeros(len(world), bool)
        suspect[:-1] |= jump
        suspect[1:] |= jump
        suspect = binary_dilation(suspect, iterations=2)
        reliable = hit & (evidence[key + "_incidence"] > .3) & (border > 5) & ~suspect & np.isfinite(predicted)
        records.append((key, xy, world, predicted, z, conf, reliable, index % 5 == 0))
    train_x = np.concatenate([r[3][r[6]] for r in records if not r[7]])
    train_y = np.concatenate([r[4][r[6]] for r in records if not r[7]])
    val_x = np.concatenate([r[3][r[6]] for r in records if r[7]])
    val_y = np.concatenate([r[4][r[6]] for r in records if r[7]])
    assert len(train_x) > 30 and len(val_x) > 30
    design = np.column_stack([train_x, np.ones(len(train_x))])
    weights = np.ones(len(train_x))
    for _ in range(10):
        coefficient = np.linalg.lstsq(design * np.sqrt(weights[:, None]), train_y * np.sqrt(weights), rcond=None)[0]
        residual = design @ coefficient - train_y
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-5)
        weights = np.minimum(1, 1.345 * scale / np.maximum(np.abs(residual), 1e-10))
    predicted_val = val_x * coefficient[0] + coefficient[1]
    errors_mm = np.abs(predicted_val - val_y) * ray_m_per_z * 1000
    baseline_mm = np.abs(np.median(train_y) - val_y) * ray_m_per_z * 1000
    passed = bool(coefficient[0] < 0 and np.quantile(errors_mm, .9) <= args.validation_p90_mm
                  and np.median(errors_mm) < np.median(baseline_mm))
    report = {"parameters": vars(args), "calibration_passed": passed,
              "criteria": "斜率为负；留出曲线p90物理误差不超过阈值；中位误差优于训练中位常量深度。网格仅为参考。",
              "train_points": len(train_x), "validation_points": len(val_x),
              "validation_guides": [r[0] for r in records if r[7]],
              "affine_scale_offset": coefficient.tolist(), "meters_per_camera_z": ray_m_per_z,
              "validation_error_mm_q50_q90": np.quantile(errors_mm, [.5,.9]).tolist(),
              "constant_baseline_error_mm_q50_q90": np.quantile(baseline_mm,[.5,.9]).tolist(),
              "notes": "按曲线留出但曲线在空间邻近，不是独立场景验证。网格首交点无逐面语义保证。20mm为本次试验验收阈值，不代表已验证精度标准。"}
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(val_x, val_y * ray_m_per_z * 1000, s=2, alpha=.2)
    axis = np.linspace(val_x.min(), val_x.max(), 100)
    axes[0].plot(axis, (axis * coefficient[0] + coefficient[1]) * ray_m_per_z * 1000, c="red")
    axes[0].set(xlabel="Depth Pro (m)", ylabel="Mesh camera-depth coordinate (mm)", title="Held-out guides: affine fit")
    axes[1].hist(errors_mm, bins=50)
    axes[1].axvline(args.validation_p90_mm,c="red",ls="--")
    axes[1].set(xlabel="Absolute depth error (mm)", title="Validation errors (mesh reference)")
    fig.tight_layout(); fig.savefig(out / "calibration_check.png",dpi=160); plt.close(fig)
    if passed:
        fused = {}
        for key, xy, world, predicted, z, conf, reliable, heldout in records:
            if reliable.sum() < 6 or not np.isfinite(predicted).all():
                continue
            n = len(xy)
            prior = predicted * coefficient[0] + coefficient[1]
            anchor_weight = reliable.astype(float) * 10
            prior_weight = .2 * conf + .001
            second = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n))
            matrix = sparse.diags(anchor_weight + prior_weight) + 20 * second.T @ second
            rhs = anchor_weight * np.nan_to_num(z) + prior_weight * prior
            solved = spsolve(matrix.tocsc(), rhs)
            uv = xy / [depth.shape[1]-1, depth.shape[0]-1] * 2 - 1
            homogeneous = np.column_stack([uv, solved, np.ones(n)]) @ inverse.T
            fused[key] = (homogeneous[:,:3] / homogeneous[:,3:4]).astype(np.float32)
        np.savez_compressed(out / "fused_guides_world.npz", **fused)
        if fused:
            save_polyline_strands(list(fused.values()), str(out / "fused_guides.ply"))
        report["fused_guides"] = len(fused)
    else:
        report["fusion_status"] = "未通过校准，停止生成融合曲线；未连接头皮或修改PDE。"
    (out / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    run_sparse_depth_fusion()
