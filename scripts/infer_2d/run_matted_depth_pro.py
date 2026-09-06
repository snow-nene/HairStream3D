"""用 seg 抠图重跑 Depth Pro，输出轮廓异常修补及置信区域。"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def run_matted_depth_pro():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--band-px", type=int, default=5)
    args = parser.parse_args()
    if args.band_px < 1:
        parser.error("band-px 必须为正")
    repo = Path(__file__).resolve().parents[2]
    vendor = repo / "ext/MVGenMaster"
    sys.path.insert(0, str(vendor))
    from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
    base = repo / "results/multiview_data" / args.image_id
    previous = base / "pde_governance/sparse_guides/step_04_depth_pro"
    out = base / "pde_governance/sparse_guides/step_05_matted_depth_pro"
    out.mkdir(parents=True, exist_ok=True)
    raw = cv2.imread(str(base / "raw_img.png"))
    mask = cv2.imread(str(base / "maps/seg/front.png"), 0)
    mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    inside = distance_transform_edt(mask)
    interior = inside > args.band_px
    assert interior.any()
    alpha = np.clip(inside / 2, 0, 1)
    rgba = np.dstack([raw, np.uint8(alpha * 255)])
    composite = np.uint8(raw * alpha[..., None] + 127 * (1 - alpha[..., None]))
    cv2.imwrite(str(out / "hair_cutout.png"), rgba)
    cv2.imwrite(str(out / "neutral_background_input.png"), composite)
    focal = json.loads((previous / "inference.json").read_text())["focal_length_px"]
    DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = str(vendor / "check_points/depth_pro.pt")
    torch.set_num_threads(8)
    start = time.time()
    print("Loading Depth Pro CPU; fixed focal:", focal, flush=True)
    model, transform = create_model_and_transforms(device=torch.device("cpu"), precision=torch.float32)
    model.eval()
    with torch.inference_mode():
        prediction = model.infer(transform(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)), f_px=torch.tensor(focal))
    predicted = prediction["depth"].cpu().numpy().squeeze()
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    # 蒙版内距边界五像素以上作为外推来源；只修补窄带中偏离来源显著的点。
    nearest = distance_transform_edt(~interior, return_distances=False, return_indices=True)
    reference = predicted[tuple(nearest)]
    band = mask & ~interior
    anomalous = band & (np.abs(predicted - reference) > np.maximum(0.15, 0.06 * reference))
    corrected = predicted.copy()
    corrected[anomalous] = reference[anomalous]
    corrected[~mask] = np.nan
    confidence = np.where(mask, np.clip(inside / (2 * args.band_px), 0, 1), 0).astype(np.float32)
    confidence[anomalous] = 0.1
    np.save(out / "matted_prediction.npy", predicted)
    np.save(out / "corrected_hair_depth.npy", corrected)
    np.save(out / "depth_confidence.npy", confidence)
    cv2.imwrite(str(out / "repaired_mask.png"), np.uint8(anomalous) * 255)
    old = np.load(previous / "depth_pro_metric.npy")
    lo, hi = np.quantile(np.concatenate([old[interior], predicted[interior]]), [.02, .98])
    panels = [composite]
    for values, title in ((old, "Original Depth Pro"), (predicted, "Matted Depth Pro"),
                           (corrected, "Boundary repaired")):
        normalized = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(hi - lo, 1e-8), 0, 1)
        color = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
        color[~mask] = 127
        cv2.putText(color, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
        panels.append(color)
    cv2.imwrite(str(out / "comparison.png"), np.concatenate(panels, axis=1))
    report = {"parameters": vars(args), "fixed_focal_px": focal,
              "elapsed_seconds": time.time() - start,
              "hair_pixels": int(mask.sum()), "boundary_pixels": int(band.sum()),
              "repaired_pixels": int(anomalous.sum()), "shared_visual_depth_range_m": [float(lo), float(hi)],
              "original_boundary_depth_q50_q98": np.quantile(old[band], [.5,.98]).tolist(),
              "matted_boundary_depth_q50_q98": np.quantile(predicted[band], [.5,.98]).tolist(),
              "corrected_boundary_depth_q50_q98": np.quantile(corrected[band], [.5,.98]).tolist(),
              "note": "seg 蒙版抠图，非精细 alpha matting；中性背景可能改变深度预测。修补为最近内部深度外推，置信度0.1；外部深度NaN。尚未与网格对齐。"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run_matted_depth_pro()
