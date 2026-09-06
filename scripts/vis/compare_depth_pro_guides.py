"""比较 Depth Pro 与网格射线的曲线深度；拟合仅用于相对深度审计。"""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

base = Path("results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888")
root = base / "pde_governance/sparse_guides"
out = root / "step_04_depth_pro"
depth = np.load(out / "depth_pro_metric.npy")
raw = cv2.imread(str(base / "raw_img.png"))
seg = cv2.imread(str(base / "maps/seg/front.png"), 0) > 127
old = np.load(base / "maps/depth_map/front.npy")
guides = np.load(root / "step_01_front_curves/guides_pixel_depth.npz")
evidence = np.load(root / "step_03_mesh_lift/raycast_evidence.npz")
metric, world_z, usable = [], [], []
for key in guides.files:
    xy = guides[key][:, :2]
    values = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1)
    points = evidence[key + "_world_raw"]
    valid = evidence[key + "_hit_mask"] & (evidence[key + "_incidence"] > 0.2)
    metric.append(values)
    world_z.append(points[:, 2])
    usable.append(valid)
x, y, mask = np.concatenate(metric), np.concatenate(world_z), np.concatenate(usable)
design = np.column_stack([x, np.ones(len(x))])
selected = mask.copy()
for iteration in range(5):
    coefficients = np.linalg.lstsq(design[selected], y[selected], rcond=None)[0]
    residual = y - design @ coefficients
    center = np.median(residual[selected])
    sigma = max(1.4826 * np.median(np.abs(residual[selected] - center)), 0.001)
    selected = mask & (np.abs(residual - center) < 2.5 * sigma)
panels = [raw]
for values, title in ((old, "Original feature depth (relative)"),
                      (depth, "Depth Pro: hair contrast")):
    lo, hi = np.quantile(values[seg], [0.02, 0.98])
    colors = cv2.applyColorMap(np.uint8(np.clip((values - lo) / max(hi-lo, 1e-8), 0, 1) * 255), cv2.COLORMAP_TURBO)
    colors[~seg] = (raw[~seg] * 0.25).astype(np.uint8)
    cv2.putText(colors, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
    panels.append(colors)
cv2.imwrite(str(out / "depth_comparison.png"), np.concatenate(panels, axis=1))
lo, hi = np.quantile(depth, [0.02, 0.98])
full = cv2.applyColorMap(np.uint8(np.clip((depth-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO)
cv2.imwrite(str(out / "depth_pro_full.png"), full)
aligned_steps, mesh_steps = [], []
for index, key in enumerate(guides.files):
    aligned_steps.extend(np.abs(np.diff(metric[index]) * coefficients[0]).tolist())
    dz = np.abs(np.diff(world_z[index]))
    mesh_steps.extend(dz[np.isfinite(dz)].tolist())
report = {
    "hair_depth_m_q02_q50_q98": np.quantile(depth[seg], [.02,.5,.98]).tolist(),
    "affine_a_b_for_world_z_equals_a_depth_plus_b": coefficients.tolist(),
    "fit_samples": int(mask.sum()), "retained_samples": int(selected.sum()),
    "all_valid_fit_abs_error_m_q50_q90": np.quantile(np.abs(residual[mask]), [.5,.9]).tolist(),
    "trimmed_fit_abs_error_m_q50_q90": np.quantile(np.abs(residual[selected]), [.5,.9]).tolist(),
    "mesh_abs_delta_world_z_m_q50_q90_q99": np.quantile(mesh_steps,[.5,.9,.99]).tolist(),
    "depth_pro_affine_abs_delta_world_z_m_q50_q90_q99": np.quantile(aligned_steps,[.5,.9,.99]).tolist(),
    "limitations": "仅全局仿射比较，不是相机标定；world_z 非严格相机深度。原特征深度与 Depth Pro 各自归一化着色，颜色不可直接比较。网格不是深度真值，平滑不能证明正确。未替换管线。",
}
(out / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
