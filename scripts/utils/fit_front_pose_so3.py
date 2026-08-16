#!/usr/bin/env python3
"""用合法 SO(3) 旋转拟合标准头模到真实 front 图像。"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.mesh_util import load_obj_mesh
from lib.util.opt_lmk import load_point_ids
from scripts.recon_3d.recon3D import load_calib
from scripts.utils.align_glb_lmk import get_lmk


def nearest_rotation(matrix):
    """Project a free 3x3 affine block onto SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def project_landmarks(points, rotation, center, scale, ortho_ratio, image_size):
    camera = (rotation @ (points - center).T).T
    ndc = camera[:, :2] * (float(scale) / float(ortho_ratio)) / 512.0
    ndc[:, 1] *= -1.0
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * image_size - 1.0
    pixels[:, 1] = (ndc[:, 1] + 1.0) * 0.5 * image_size - 1.0
    return pixels


def landmark_weights(count):
    """Emphasize pose-stable inner-face landmarks over jaw/hair occlusions."""
    weights = np.ones(count, dtype=np.float64)
    weights[:17] = 0.20
    weights[17:27] = 0.65
    weights[27:36] = 1.50
    weights[36:48] = 1.25
    weights[48:68] = 0.85
    return weights


def build_param(rotation, center, scale, ortho_ratio):
    return {
        "ortho_ratio": float(ortho_ratio),
        "scale": np.asarray([scale], dtype=np.float32),
        "center": np.asarray(center, dtype=np.float32).reshape(3, 1),
        "R": np.asarray(rotation, dtype=np.float32),
    }


def render_diagnostic(image, mesh, calib, gt_lmk, pred_lmk):
    height, width = image.shape[:2]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices), np.float32)])
    projected = homogeneous @ calib.T
    px = np.rint((projected[:, 0] + 1.0) * 0.5 * (width - 1)).astype(int)
    py = np.rint((projected[:, 1] + 1.0) * 0.5 * (height - 1)).astype(int)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    canvas = image.copy()
    for x, y in zip(px[inside][::4], py[inside][::4]):
        cv2.circle(canvas, (x, y), 1, (0, 220, 0), -1)
    for point in gt_lmk:
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (0, 0, 255), -1)
    for point in pred_lmk:
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (255, 80, 0), -1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    data_dir = ROOT / "results" / "multiview_data" / args.img_id
    if args.image:
        image_path = Path(args.image)
    else:
        raw_path_file = data_dir / "raw_img_path.txt"
        image_path = Path(raw_path_file.read_text(encoding="utf-8").strip())
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_dir / "pde_smoothness_experiments" / "front_pose_so3"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    detected = get_lmk(image_512)
    if detected is None or len(detected) < 68:
        raise RuntimeError("无法在 front 图像中检测到 68 个人脸关键点")
    target = detected[:68, :2].astype(np.float64)

    head_vertices, _ = load_obj_mesh(str(ROOT / "data" / "head_model.obj"))
    landmark_ids = load_point_ids(str(ROOT / "data" / "landmark_id_uschair.obj"))
    source = np.asarray(head_vertices[landmark_ids[:68]], dtype=np.float64)

    old_path = data_dir / "maps" / "param" / "front.npy"
    old_param = np.load(old_path, allow_pickle=True).item()
    ortho_ratio = float(old_param.get("ortho_ratio", 0.2))
    initial_rotation = nearest_rotation(old_param["R"])
    initial_center = np.asarray(old_param["center"], dtype=np.float64).reshape(3)
    initial_scale = float(np.asarray(old_param["scale"]).reshape(-1)[0])
    weights = landmark_weights(len(source))

    old_camera_depth = float(
        (initial_rotation @ initial_center.reshape(3, 1))[2, 0]
    )

    def unpack(values):
        rotation = Rotation.from_rotvec(values[:3]).as_matrix()
        center = np.array([values[3], values[4], 0.0], dtype=np.float64)
        # Orthographic landmarks contain no depth information. Preserve the
        # old camera-space center depth so the front depth-map gate remains
        # numerically compatible after changing rotation.
        center[2] = (
            old_camera_depth
            - rotation[2, 0] * center[0]
            - rotation[2, 1] * center[1]
        ) / rotation[2, 2]
        scale = np.exp(values[5])
        return rotation, center, scale

    def residual(values):
        rotation, center, scale = unpack(values)
        predicted = project_landmarks(
            source, rotation, center, scale, ortho_ratio, 512
        )
        return ((predicted - target) * np.sqrt(weights[:, None])).ravel()

    base_rotvec = Rotation.from_matrix(initial_rotation).as_rotvec()
    candidates = []
    for yaw in (-15.0, 0.0, 15.0):
        delta = Rotation.from_euler("y", yaw, degrees=True)
        start_rotation = delta * Rotation.from_rotvec(base_rotvec)
        x0 = np.r_[
            start_rotation.as_rotvec(), initial_center[:2], np.log(initial_scale)
        ]
        result = least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        candidates.append(result)
    best = min(candidates, key=lambda item: np.mean(np.abs(residual(item.x))))
    rotation, center, scale = unpack(best.x)
    predicted = project_landmarks(
        source, rotation, center, scale, ortho_ratio, 512
    )

    param = build_param(rotation, center, scale, ortho_ratio)
    param_path = out_dir / "front_so3.npy"
    np.save(param_path, param)
    calib = load_calib(str(param_path), loadSize=1024).numpy()
    mesh = o3d.io.read_triangle_mesh(str(ROOT / "data" / "head_model.obj"))
    diagnostic = render_diagnostic(image_512, mesh, calib, target, predicted)
    cv2.imwrite(str(out_dir / "head_pose_overlay.png"), diagnostic)

    errors = np.linalg.norm(predicted - target, axis=1)
    report = (
        f"mean_error_px={errors.mean():.4f}\n"
        f"median_error_px={np.median(errors):.4f}\n"
        f"inner_face_mean_px={errors[27:68].mean():.4f}\n"
        f"det_R={np.linalg.det(rotation):.8f}\n"
        f"orthogonality_error={np.linalg.norm(rotation.T @ rotation - np.eye(3)):.8e}\n"
        f"euler_xyz_deg={Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist()}\n"
        f"scale={scale:.8f}\n"
        f"center={center.tolist()}\n"
    )
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"param={param_path}")
    print(f"overlay={out_dir / 'head_pose_overlay.png'}")


if __name__ == "__main__":
    main()
