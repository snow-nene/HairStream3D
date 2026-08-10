"""Silhouette 贴合度评测：把重建发丝 PLY 投影回输入 front 图像并对比 seg。

复现指标：可见轮廓面积、IoU、输入轮廓召回率、重建轮廓精度、bbox 对比。
投影使用真实 front 标定 (maps/param/front.npy)，并用头模做遮挡测试，
与重建管线使用完全一致的相机约定。

用法::

    pixi run python tests/test_silhouette_iou.py --img_id <id> [--ply path]
        [--label name] [--dilate_px 0]

输出统一写入 tests/outputs/silhouette_iou/<img_id>/。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d


def bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def evaluate_ply(ply_path, guard, seg, label, out_dir, dilate_px=0):
    line_set = o3d.io.read_line_set(str(ply_path))
    points = np.asarray(line_set.points, dtype=np.float32)
    lines = np.asarray(line_set.lines)
    print(f"[{label}] {ply_path}: {len(points)} points, {len(lines)} segments")

    visible = guard._visible(guard.primary, points)
    px, py = guard._project_px(guard.primary, points)
    height, width = seg.shape
    in_img = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    ok = in_img & visible

    # 按 3D 线段光栅化（与渲染轮廓一致），跳过跨越遮挡边界的线段
    canvas = np.zeros((height, width), dtype=np.uint8)
    a = lines[:, 0]
    b = lines[:, 1]
    keep = ok[a] & ok[b]
    pa = np.stack([px[a[keep]], py[a[keep]]], axis=1).astype(np.int32)
    pb = np.stack([px[b[keep]], py[b[keep]]], axis=1).astype(np.int32)
    for start, end in zip(pa, pb):
        cv2.line(canvas, tuple(start), tuple(end), 1, thickness=1)
    canvas = canvas > 0
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        canvas = cv2.dilate(canvas.astype(np.uint8), kernel) > 0

    inter = np.logical_and(canvas, seg).sum()
    union = np.logical_or(canvas, seg).sum()
    iou = inter / max(union, 1)
    recall = inter / max(seg.sum(), 1)
    precision = inter / max(canvas.sum(), 1)
    seg_box = bbox_of_mask(seg)
    rec_box = bbox_of_mask(canvas)

    report = {
        "label": label,
        "ply": str(ply_path),
        "seg_area_px": int(seg.sum()),
        "recon_visible_area_px": int(canvas.sum()),
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "seg_bbox": seg_box,
        "recon_bbox": rec_box,
    }
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 可视化：seg=红，重建可见轮廓=绿，重叠=黄
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[seg] = (255, 0, 0)
    overlay[canvas] = (0, 255, 0)
    overlay[seg & canvas] = (255, 255, 0)
    vis_path = os.path.join(out_dir, f"overlay_{label}.png")
    imageio.imwrite(vis_path, overlay)
    print(f"  overlay -> {vis_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Silhouette IoU evaluation")
    parser.add_argument("--img_id", required=True)
    parser.add_argument(
        "--ply",
        nargs="*",
        default=None,
        help="一个或多个 PLY；默认 pde_reconstruction/hair_multiview.ply",
    )
    parser.add_argument("--label", nargs="*", default=None)
    parser.add_argument("--dilate_px", type=int, default=0)
    args = parser.parse_args()

    from lib.silhouette_guard import SilhouetteGuard
    from scripts.recon_3d.recon3D import load_calib

    data_dir = os.path.join(ROOT, "results", "multiview_data", args.img_id)
    seg_path = os.path.join(data_dir, "maps", "seg", "front.png")
    calib_path = os.path.join(data_dir, "maps", "param", "front.npy")
    seg = imageio.imread(seg_path)
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    seg = seg > 127
    calib = load_calib(calib_path, loadSize=1024)
    if hasattr(calib, "numpy"):
        calib = calib.numpy()

    guard = SilhouetteGuard(
        {"front": seg},
        {"front": (calib, np.eye(3, dtype=np.float32))},
        head_mesh_path=os.path.join(ROOT, "data", "head_model.obj"),
        hard_px=0.0,
        primary_view="front",
    )

    plys = args.ply or [
        os.path.join(data_dir, "pde_reconstruction", "hair_multiview.ply")
    ]
    labels = args.label or [
        os.path.splitext(os.path.basename(p))[0] for p in plys
    ]

    out_dir = os.path.join(ROOT, "tests", "outputs", "silhouette_iou", args.img_id)
    os.makedirs(out_dir, exist_ok=True)

    reports = [
        evaluate_ply(p, guard, seg, lab, out_dir, args.dilate_px)
        for p, lab in zip(plys, labels)
    ]

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "a", encoding="utf-8") as handle:
        import datetime

        handle.write(
            f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} ===\n"
        )
        for report in reports:
            handle.write(
                f"{report['label']}: IoU={report['iou']:.4f} "
                f"recall={report['recall']:.4f} "
                f"precision={report['precision']:.4f} "
                f"area(seg/recon)={report['seg_area_px']}/{report['recon_visible_area_px']} "
                f"bbox(seg)={report['seg_bbox']} bbox(recon)={report['recon_bbox']} "
                f"ply={report['ply']}\n"
            )
    print(f"\nreport appended -> {report_path}")


if __name__ == "__main__":
    main()
