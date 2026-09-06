"""从正面 strand/depth 提取稀疏可见曲线，输出原图叠加供人工验收。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.multiview_fusion import trace_view_strands_3d


def run_sparse_guide_audit():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--max-guides", type=int, default=200)
    parser.add_argument("--depth-jump", type=float, default=0.03,
                        help="每个追踪步的归一化深度差阈值，不是米")
    args = parser.parse_args()
    if args.max_guides < 1 or args.depth_jump <= 0:
        parser.error("max-guides 和 depth-jump 必须为正")
    base = Path("results/multiview_data") / args.image_id
    out = base / "pde_governance/sparse_guides/step_01_front_curves"
    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        "raw": base / "raw_img.png",
        "strand": base / "maps/strand_map/front.png",
        "depth": base / "maps/depth_map/front.npy",
        "seg": base / "maps/seg/front.png",
        "parting": base / "pde_governance/front_parting_dinov3_migrated/parting_region_mask.png",
    }
    raw = cv2.imread(str(inputs["raw"]))
    strand = cv2.cvtColor(cv2.imread(str(inputs["strand"])), cv2.COLOR_BGR2RGB) / 255.0
    depth = np.load(inputs["depth"])
    height, width = depth.shape
    seg = cv2.imread(str(inputs["seg"]), cv2.IMREAD_GRAYSCALE)
    parting = cv2.imread(str(inputs["parting"]), cv2.IMREAD_GRAYSCALE)
    seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST) > 127
    parting = cv2.resize(parting, (width, height), interpolation=cv2.INTER_NEAREST) > 127
    assert strand.shape == (height, width, 3)
    valid = seg & ~parting & np.isfinite(depth) & (depth > 0.05)
    strand = strand.astype(np.float32)
    strand[~valid] = 0
    clean_depth = np.where(valid, depth, 0).astype(np.float32)
    # 单位标定仅用于取回像素曲线；此处不生成世界坐标三维毛发。
    traced = trace_view_strands_3d(
        strand, clean_depth, np.eye(4), seed_spacing=4,
        step_pixels=1.0, max_steps=512, min_curve_points=12,
    )
    candidates = []
    split_points = []
    reasons = {"depth": 0, "turn": 0, "mask": 0, "loop": 0}
    for curve in traced:
        xy = (curve[:, :2] + 1) * np.array([width - 1, height - 1]) / 2
        pixels = np.rint(xy).astype(int)
        good = valid[pixels[:, 1], pixels[:, 0]]
        z = map_coordinates(depth, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        jumps = np.abs(np.diff(z)) > args.depth_jump
        delta = np.diff(xy, axis=0)
        tangents = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
        turns = np.zeros(len(xy) - 1, dtype=bool)
        turns[1:] = np.sum(tangents[1:] * tangents[:-1], axis=1) < np.cos(np.deg2rad(45))
        badmask = ~good[1:] | ~good[:-1]
        reasons["depth"] += int(jumps.sum())
        reasons["turn"] += int(turns.sum())
        reasons["mask"] += int(badmask.sum())
        cuts = np.flatnonzero(jumps | turns | badmask) + 1
        split_points.extend(xy[cuts].tolist())
        for indices in np.split(np.arange(len(xy)), cuts):
            if len(indices) < 20 or not good[indices].all():
                continue
            pts = xy[indices]
            # 拒绝回到十步以前位置附近的自循环尾段。
            end = len(pts)
            for index in range(12, len(pts)):
                if np.min(np.linalg.norm(pts[:index - 10] - pts[index], axis=1)) < 2:
                    end = index
                    reasons["loop"] += 1
                    break
            pts = pts[:end]
            if len(pts) >= 20:
                candidates.append((pts, z[indices][:end]))
    candidates.sort(key=lambda item: -len(item[0]))
    covered = np.zeros((height, width), np.uint8)
    selected = []
    for pts, z in candidates:
        pixels = np.rint(pts).astype(int)
        if np.mean(covered[pixels[:, 1], pixels[:, 0]] == 0) < 0.35:
            continue
        selected.append((pts, z))
        cv2.polylines(covered, [pixels], False, 1, 5)
        if len(selected) >= args.max_guides:
            break
    assert selected, "没有足够长的有效曲线"
    canvas = raw.copy()
    depth_canvas = raw.copy()
    scale = np.array([raw.shape[1] / width, raw.shape[0] / height])
    low, high = np.quantile(depth[valid], [0.02, 0.98])
    records = {}
    for index, (pts, z) in enumerate(selected):
        color = cv2.cvtColor(np.uint8([[[index * 47 % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        pixels = np.rint(pts * scale).astype(np.int32)
        cv2.polylines(canvas, [pixels], False, tuple(map(int, color)), 1, cv2.LINE_AA)
        for endpoint in (pixels[0], pixels[-1]):
            cv2.circle(canvas, tuple(endpoint), 2, (255, 255, 255), -1)
        value = np.uint8(np.clip((np.median(z) - low) / max(high - low, 1e-8), 0, 1) * 255)
        depth_color = cv2.applyColorMap(np.array([[value]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        cv2.polylines(depth_canvas, [pixels], False, tuple(map(int, depth_color)), 1, cv2.LINE_AA)
        records[f"guide_{index:03d}"] = np.column_stack([pts, z]).astype(np.float32)
    cv2.imwrite(str(out / "guides_on_raw.png"), canvas)
    cv2.imwrite(str(out / "guides_relative_depth_on_raw.png"), depth_canvas)
    cv2.imwrite(str(out / "comparison.png"), np.concatenate([raw, canvas, depth_canvas], axis=1))
    np.savez_compressed(out / "guides_pixel_depth.npz", **records)
    report = {
        "说明": "可见无向曲线片段，非根到梢完整发丝；深度为预测相对深度，颜色不代表已确认分层。",
        "inputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()},
        "parameters": vars(args), "raw_traced_curves": len(traced),
        "eligible_fragments": len(candidates), "selected_guides": len(selected),
        "split_events": reasons,
        "length_pixels_quantiles": np.quantile([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p, _ in selected], [0.1, 0.5, 0.9]).tolist(),
        "mask_coverage_with_5px_stroke": float(np.mean(covered[valid] > 0)),
        "depth_color_range": [float(low), float(high)],
        "checks": {"all_samples_inside_mask": True, "min_fragment_points": 20},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(out / "comparison.png")


if __name__ == "__main__":
    run_sparse_guide_audit()
