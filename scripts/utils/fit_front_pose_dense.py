#!/usr/bin/env python3
"""用渲染图与原图的稠密 DINO 特征拟合 front 正交相机。

渲染侧像素会通过 Blender 相机射线与原始 GLB 求交，因此优化使用的源点
是真实可见表面点，而不是把二维点放在一个近似深度平面上。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import timm
import torch
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recon_3d.recon3D import load_calib


MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
FEATURE_SIZE = 512
PATCH_SIZE = 16


def nearest_rotation(matrix):
    """将旧标定中的自由 3x3 矩阵投影到合法 SO(3)。"""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def extract_dino_features(image, model, device):
    """提取 512x512 图像的 32x32 DINOv3 patch 特征。"""
    resized = cv2.resize(image, (FEATURE_SIZE, FEATURE_SIZE), cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(
        model.pretrained_cfg.get("mean", (0.485, 0.456, 0.406)),
        dtype=np.float32,
    )
    std = np.asarray(
        model.pretrained_cfg.get("std", (0.229, 0.224, 0.225)),
        dtype=np.float32,
    )
    tensor = torch.from_numpy((rgb - mean) / std).permute(2, 0, 1)[None]
    tensor = tensor.to(device)
    with torch.inference_mode():
        tokens = model.forward_features(tensor)
        tokens = tokens[:, model.num_prefix_tokens :]
        tokens = torch.nn.functional.normalize(tokens.float(), dim=-1)
    grid = FEATURE_SIZE // PATCH_SIZE
    return tokens[0].cpu().numpy().reshape(grid, grid, -1), resized


def elliptical_patch_ids(grid):
    """保留头部主体，排除背景和肩部；不依赖人脸 landmark。"""
    yy, xx = np.mgrid[:grid, :grid]
    x = xx + 0.5
    y = yy + 0.5
    mask = ((x - grid / 2) / (0.375 * grid)) ** 2
    mask += ((y - 0.47 * grid) / (0.453 * grid)) ** 2
    mask = (mask < 1.0) & (y < 0.845 * grid)
    return np.flatnonzero(mask.ravel())


def match_patch_features(render_features, target_features):
    """互为最近邻匹配，并用单一全局相似变换剔除离群对应。"""
    grid = render_features.shape[0]
    render_ids = elliptical_patch_ids(grid)
    target_ids = elliptical_patch_ids(grid)
    render_flat = render_features.reshape(-1, render_features.shape[-1])
    target_flat = target_features.reshape(-1, target_features.shape[-1])
    similarity = render_flat[render_ids] @ target_flat[target_ids].T

    target_nn = similarity.argmax(axis=1)
    render_nn = similarity.argmax(axis=0)
    rows = np.arange(len(render_ids))
    mutual = rows == render_nn[target_nn]
    score = similarity[rows, target_nn]
    top_two = np.partition(similarity, -2, axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    keep = mutual & (score > 0.70) & (margin > 0.005)

    matched_render_ids = render_ids[keep]
    matched_target_ids = target_ids[target_nn[keep]]
    render_xy = np.column_stack(
        [matched_render_ids % grid + 0.5, matched_render_ids // grid + 0.5]
    ) * PATCH_SIZE
    target_xy = np.column_stack(
        [matched_target_ids % grid + 0.5, matched_target_ids // grid + 0.5]
    ) * PATCH_SIZE
    scores = score[keep]
    if len(render_xy) < 12:
        raise RuntimeError(f"有效 DINO 匹配过少: {len(render_xy)}")

    affine, inliers = cv2.estimateAffinePartial2D(
        render_xy,
        target_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=32.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is None or inliers is None:
        raise RuntimeError("DINO 匹配无法形成一致的二维几何关系")
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < 10:
        raise RuntimeError(f"RANSAC 内点过少: {inliers.sum()}")
    return (
        render_xy[inliers],
        target_xy[inliers],
        scores[inliers],
        affine,
        int(mutual.sum()),
    )


def render_pixels_to_gltf_surface(render_xy, camera, mesh):
    """将渲染像素沿正交相机射线投到原始 GLB 的首个可见表面。"""
    size = float(camera["img_size"])
    ortho_scale = float(camera["ortho_scale"])
    cam_rotation = np.asarray(camera["cam_R"], dtype=np.float64)
    cam_position = np.asarray(camera["cam_pos"], dtype=np.float64)

    lx = (render_xy[:, 0] - size / 2.0) / (size / 2.0) * (ortho_scale / 2.0)
    ly = -(render_xy[:, 1] - size / 2.0) / (size / 2.0) * (ortho_scale / 2.0)
    local_origins = np.column_stack([lx, ly, np.zeros(len(lx))])
    origins_blender = local_origins @ cam_rotation.T + cam_position
    direction_blender = -cam_rotation[:, 2]

    origins_gltf = np.column_stack(
        [origins_blender[:, 0], origins_blender[:, 2], -origins_blender[:, 1]]
    )
    direction_gltf = np.asarray(
        [direction_blender[0], direction_blender[2], -direction_blender[1]],
        dtype=np.float64,
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.column_stack(
        [origins_gltf, np.tile(direction_gltf, (len(origins_gltf), 1))]
    )
    result = scene.cast_rays(o3d.core.Tensor(rays, o3d.core.Dtype.Float32))
    hit_distance = result["t_hit"].numpy().astype(np.float64)
    valid = np.isfinite(hit_distance) & (hit_distance < 1e5)
    points = origins_gltf + hit_distance[:, None] * direction_gltf
    return points, valid


def gltf_to_world(points, transform_path):
    """应用 align_and_extract_hair 保存的 GLB 到 head_model 世界变换。"""
    transform = np.load(transform_path)
    scale = float(np.asarray(transform["umeyama_scale"]).reshape(-1)[0])
    rotation = np.asarray(transform["umeyama_R"], dtype=np.float64)
    translation = np.asarray(transform["umeyama_t"], dtype=np.float64)
    icp = np.asarray(transform["icp_T"], dtype=np.float64)
    aligned = scale * (points @ rotation.T) + translation
    homogeneous = np.column_stack([aligned, np.ones(len(aligned))])
    return (homogeneous @ icp.T)[:, :3]


def gltf_vertices_to_world(vertices, transform_path):
    """批量转换 GLB 顶点，保留原始面拓扑供轮廓渲染。"""
    return gltf_to_world(np.asarray(vertices, dtype=np.float64), transform_path)


def select_render_hair_faces(vertices, faces, camera, render_hair_mask):
    """用渲染图头发 mask 选择原始 GLB 中可见的头发表面。"""
    centroids = vertices[faces].mean(axis=1)
    blender = np.column_stack(
        [centroids[:, 0], -centroids[:, 2], centroids[:, 1]]
    )
    cam_rotation = np.asarray(camera["cam_R"], dtype=np.float64)
    cam_position = np.asarray(camera["cam_pos"], dtype=np.float64)
    local = (blender - cam_position) @ cam_rotation
    size = int(camera["img_size"])
    ortho_scale = float(camera["ortho_scale"])
    px = np.rint(local[:, 0] / (ortho_scale / 2.0) * (size / 2.0)
                 + size / 2.0).astype(np.int32)
    py = np.rint(-local[:, 1] / (ortho_scale / 2.0) * (size / 2.0)
                 + size / 2.0).astype(np.int32)
    mask = cv2.resize(
        render_hair_mask, (size, size), interpolation=cv2.INTER_NEAREST
    )
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    selected = np.zeros(len(faces), dtype=bool)
    selected[inside] = mask[py[inside], px[inside]] > 127
    return selected


def rasterize_silhouette(vertices, faces, param, image_size=512):
    """将选中的三角形并集光栅化为正交轮廓。"""
    param_path_scale = float(np.asarray(param["scale"]).reshape(-1)[0])
    ortho_ratio = float(param["ortho_ratio"])
    rotation = np.asarray(param["R"], dtype=np.float64)
    center = np.asarray(param["center"], dtype=np.float64).reshape(3)
    camera = (vertices - center) @ rotation.T
    ndc = camera[:, :2] * (param_path_scale / ortho_ratio) / 512.0
    ndc[:, 1] *= -1.0
    pixels = np.column_stack(
        [(ndc[:, 0] + 1.0) * 0.5 * image_size - 1.0,
         (ndc[:, 1] + 1.0) * 0.5 * image_size - 1.0]
    )
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, np.rint(pixels[faces]).astype(np.int32), 255)
    return mask


def refine_with_hair_silhouette(source_mask, target_mask):
    """用头发轮廓 IoU 优化小范围二维相似变换。"""
    height, width = target_mask.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    target = target_mask > 127

    def matrix(values):
        scale = np.exp(values[0])
        affine = cv2.getRotationMatrix2D(center, values[1], scale)
        affine[:, 2] += values[2:4]
        return affine

    def warped(values):
        return cv2.warpAffine(
            source_mask,
            matrix(values),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 127

    def objective(values):
        candidate = warped(values)
        intersection = np.logical_and(candidate, target).sum()
        union = np.logical_or(candidate, target).sum()
        return 1.0 - intersection / max(int(union), 1)

    result = differential_evolution(
        objective,
        [
            (np.log(0.85), np.log(1.08)),
            (-4.0, 4.0),
            (-30.0, 30.0),
            (-30.0, 30.0),
        ],
        seed=2,
        popsize=12,
        maxiter=60,
        tol=1e-6,
        polish=True,
        workers=1,
    )
    source = source_mask > 127
    initial_iou = np.logical_and(source, target).sum() / max(
        int(np.logical_or(source, target).sum()), 1
    )
    return result.x, matrix(result.x), warped(result.x), initial_iou, 1.0 - result.fun


def apply_image_similarity(rotation, ppw, tx, ty, values, affine):
    """把 OpenCV 图像相似变换吸收到正交相机参数中。"""
    scale = float(np.exp(values[0]))
    angle = float(values[1])
    roll = Rotation.from_euler("z", angle, degrees=True).as_matrix()
    refined_rotation = roll @ rotation
    refined_translation = affine[:, :2] @ np.asarray([tx, ty]) + affine[:, 2]
    return (
        refined_rotation,
        ppw * scale,
        float(refined_translation[0]),
        float(refined_translation[1]),
    )


def project_points(points, rotation, pixels_per_world, tx, ty):
    camera = points @ rotation.T
    return np.column_stack(
        [pixels_per_world * camera[:, 0] + tx,
         -pixels_per_world * camera[:, 1] + ty]
    )


def fit_camera(points, target_xy, scores, old_param):
    """以鲁棒损失拟合 SO(3)、统一尺度和二维平移。"""
    initial_rotation = nearest_rotation(old_param["R"])
    ortho_ratio = float(old_param.get("ortho_ratio", 0.2))
    old_scale = float(np.asarray(old_param["scale"]).reshape(-1)[0])
    old_center = np.asarray(old_param["center"], dtype=np.float64).reshape(3)
    initial_pixels_per_world = old_scale / (2.0 * ortho_ratio)
    old_camera_center = initial_rotation @ old_center
    initial_tx = 255.0 - initial_pixels_per_world * old_camera_center[0]
    initial_ty = 255.0 + initial_pixels_per_world * old_camera_center[1]
    weights = np.sqrt(np.clip(scores, 0.70, 1.0) / np.median(scores))

    def unpack(values):
        return (
            Rotation.from_rotvec(values[:3]).as_matrix(),
            np.exp(values[3]),
            values[4],
            values[5],
        )

    def data_residual(values, selected=None):
        rotation, ppw, tx, ty = unpack(values)
        source = points if selected is None else points[selected]
        target = target_xy if selected is None else target_xy[selected]
        weight = weights if selected is None else weights[selected]
        return ((project_points(source, rotation, ppw, tx, ty) - target)
                * weight[:, None]).ravel()

    def residual(values, selected=None):
        rotation, ppw, _, _ = unpack(values)
        data = data_residual(values, selected)
        rotation_prior = Rotation.from_matrix(
            rotation @ initial_rotation.T
        ).as_rotvec() * 12.0
        scale_prior = np.asarray(
            [np.log(ppw / initial_pixels_per_world) * 10.0]
        )
        return np.r_[data, rotation_prior, scale_prior]

    base = Rotation.from_matrix(initial_rotation)
    candidates = []
    for yaw in (-15.0, 0.0, 15.0):
        start_rotation = Rotation.from_euler("y", yaw, degrees=True) * base
        x0 = np.r_[
            start_rotation.as_rotvec(),
            np.log(initial_pixels_per_world),
            initial_tx,
            initial_ty,
        ]
        result = least_squares(
            residual,
            x0,
            loss="soft_l1",
            f_scale=5.0,
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        candidates.append(result)
    best = min(candidates, key=lambda item: np.median(np.abs(data_residual(item.x))))

    error = np.linalg.norm(
        project_points(points, *unpack(best.x)) - target_xy,
        axis=1,
    )
    selected = error < max(10.0, float(np.quantile(error, 0.80)))
    if selected.sum() >= 10 and selected.sum() < len(points):
        best = least_squares(
            lambda values: residual(values, selected),
            best.x,
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
    rotation, ppw, tx, ty = unpack(best.x)
    predicted = project_points(points, rotation, ppw, tx, ty)
    errors = np.linalg.norm(predicted - target_xy, axis=1)
    return rotation, ppw, tx, ty, predicted, errors


def build_param(rotation, pixels_per_world, tx, ty, old_param):
    """把直接像素模型转换成 HairStep front.npy 的参数格式。"""
    ortho_ratio = float(old_param.get("ortho_ratio", 0.2))
    old_center = np.asarray(old_param["center"], dtype=np.float64).reshape(3)
    center_camera = np.empty(3, dtype=np.float64)
    center_camera[0] = (255.0 - tx) / pixels_per_world
    center_camera[1] = (ty - 255.0) / pixels_per_world
    center_camera[2] = (rotation @ old_center)[2]
    center = rotation.T @ center_camera
    return {
        "ortho_ratio": ortho_ratio,
        "scale": np.asarray([2.0 * pixels_per_world * ortho_ratio], np.float32),
        "center": center.astype(np.float32).reshape(3, 1),
        "R": rotation.astype(np.float32),
    }


def draw_matches(render_image, target_image, render_xy, target_xy, scores):
    left = cv2.resize(render_image, (FEATURE_SIZE, FEATURE_SIZE), cv2.INTER_AREA)
    right = cv2.resize(target_image, (FEATURE_SIZE, FEATURE_SIZE), cv2.INTER_AREA)
    canvas = np.concatenate([left, right], axis=1)
    order = np.argsort(scores)
    for index in order:
        p0 = tuple(np.rint(render_xy[index]).astype(int))
        p1 = tuple(np.rint(target_xy[index]).astype(int) + [FEATURE_SIZE, 0])
        color = tuple(int(value) for value in cv2.applyColorMap(
            np.asarray([[int(np.clip(scores[index], 0.7, 1.0) - 0.7) / 0.3 * 255]],
                       dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )[0, 0])
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1)
        cv2.circle(canvas, p1, 2, color, -1)
    return canvas


def draw_reprojection(image, target_xy, predicted_xy, errors):
    canvas = cv2.resize(image, (FEATURE_SIZE, FEATURE_SIZE), cv2.INTER_AREA)
    for target, predicted, error in zip(target_xy, predicted_xy, errors):
        target_i = tuple(np.rint(target).astype(int))
        predicted_i = tuple(np.rint(predicted).astype(int))
        color = (0, 220, 0) if error < 6.0 else (0, 180, 255)
        cv2.line(canvas, target_i, predicted_i, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, target_i, 2, (255, 80, 0), -1)
        cv2.circle(canvas, predicted_i, 2, color, -1)
    return canvas


def generate_render_hair_mask(
    render_path,
    output_path,
    sam3_root,
    checkpoint_path,
    device,
):
    """调用项目现有 SAM3 worker，为 Pixal3D 正面渲染生成头发 mask。"""
    sam3_root = Path(sam3_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    sam3_python = sam3_root / ".pixi" / "envs" / "default" / "bin" / "python"
    worker = ROOT / "scripts" / "infer_2d" / "sam3_seg_worker.py"
    if not sam3_python.is_file():
        raise FileNotFoundError(f"缺少 SAM3 Python 环境: {sam3_python}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"缺少 SAM3 checkpoint: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / "sam3_render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "input": str(Path(render_path).resolve()),
                        "seg_out": str(output_path.resolve()),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        str(sam3_python),
        str(worker),
        "--manifest",
        str(manifest_path),
        "--checkpoint",
        str(checkpoint_path),
        "--hair_prompts",
        "hair",
        "ponytail hair and twin tails",
        "bangs and hair on the front",
        "--body_prompt",
        "person",
        "--person_selection",
        "largest",
        "--output_size",
        str(FEATURE_SIZE),
        "--mask_channels",
        "1",
        "--device",
        device,
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        manifest_path.unlink(missing_ok=True)
    if not output_path.is_file():
        raise RuntimeError(f"SAM3 未生成渲染头发 mask: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--render_hair_mask",
        default=None,
        help="可选的 Pixal3D 渲染头发 mask；存在时用 front seg 细化轮廓",
    )
    parser.add_argument(
        "--auto_render_hair_mask",
        action="store_true",
        help="render_hair_mask 不存在时调用项目 SAM3 worker 自动生成",
    )
    parser.add_argument(
        "--require_silhouette",
        action="store_true",
        help="未能完成头发轮廓细化时以非零状态退出",
    )
    parser.add_argument("--sam3_root", default=str(ROOT / "ext" / "sam3"))
    parser.add_argument(
        "--checkpoint_sam3",
        default=str(
            ROOT
            / "ext"
            / "sam3"
            / "checkpoints"
            / "facebook"
            / "sam3.1"
            / "sam3.1_multiplex.pt"
        ),
    )
    args = parser.parse_args()

    data_dir = ROOT / "results" / "multiview_data" / args.img_id
    pixal_dir = data_dir / "pixal3d"
    if args.image:
        image_path = Path(args.image)
    else:
        image_path = Path(
            (data_dir / "raw_img_path.txt").read_text(encoding="utf-8").strip()
        )
    out_dir = Path(args.out_dir) if args.out_dir else (
        data_dir / "maps" / "param" / "dense_front"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    render_path = pixal_dir / "render_front.png"
    glb_path = pixal_dir / f"{args.img_id}.glb"
    render_image = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    target_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if render_image is None or target_image is None:
        raise FileNotFoundError(f"无法读取渲染图或原图: {render_path}, {image_path}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.eval().to(device)
    render_features, _ = extract_dino_features(render_image, model, device)
    target_features, _ = extract_dino_features(target_image, model, device)
    render_xy_512, target_xy, scores, affine, mutual_count = match_patch_features(
        render_features, target_features
    )

    camera = np.load(pixal_dir / "render_camera.npz")
    render_size = float(camera["img_size"])
    render_xy = render_xy_512 * (render_size / FEATURE_SIZE)
    raw_mesh = o3d.io.read_triangle_mesh(str(glb_path))
    if not raw_mesh.has_vertices() or not raw_mesh.has_triangles():
        raise ValueError(f"无法读取有效 GLB 网格: {glb_path}")
    surface_points, valid = render_pixels_to_gltf_surface(
        render_xy, camera, raw_mesh
    )
    surface_points = gltf_to_world(
        surface_points[valid], pixal_dir / "glb_to_world.npz"
    )
    render_xy_512 = render_xy_512[valid]
    target_xy = target_xy[valid]
    scores = scores[valid]
    if len(surface_points) < 10:
        raise RuntimeError(f"GLB 表面射线交点过少: {len(surface_points)}")

    old_param_path = data_dir / "maps" / "param" / "front.npy"
    old_param = np.load(old_param_path, allow_pickle=True).item()
    rotation, ppw, tx, ty, predicted, errors = fit_camera(
        surface_points, target_xy, scores, old_param
    )
    param = build_param(rotation, ppw, tx, ty, old_param)
    param_path = out_dir / "front_dense.npy"
    np.save(param_path, param)

    if args.render_hair_mask:
        render_mask_path = Path(args.render_hair_mask)
    else:
        render_mask_path = out_dir / "render_hair_mask.png"
    if not render_mask_path.exists() and args.auto_render_hair_mask:
        worker_device = "cuda" if device == "cuda" else "cpu"
        print(f"正在生成 Pixal3D 渲染头发 mask: {render_mask_path}")
        generate_render_hair_mask(
            render_path,
            render_mask_path,
            args.sam3_root,
            args.checkpoint_sam3,
            worker_device,
        )
    target_mask_path = data_dir / "maps" / "seg" / "front.png"
    silhouette_report = None
    refined_param_path = None
    if render_mask_path.exists() and target_mask_path.exists():
        render_hair_mask = cv2.imread(str(render_mask_path), cv2.IMREAD_GRAYSCALE)
        target_hair_mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
        raw_vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
        raw_faces = np.asarray(raw_mesh.triangles, dtype=np.int32)
        hair_faces = select_render_hair_faces(
            raw_vertices, raw_faces, camera, render_hair_mask
        )
        world_vertices = gltf_vertices_to_world(
            raw_vertices, pixal_dir / "glb_to_world.npz"
        )
        source_mask = rasterize_silhouette(
            world_vertices, raw_faces[hair_faces], param, FEATURE_SIZE
        )
        target_hair_mask = cv2.resize(
            target_hair_mask,
            (FEATURE_SIZE, FEATURE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        values, affine_refine, refined_mask, iou_before, iou_after = (
            refine_with_hair_silhouette(source_mask, target_hair_mask)
        )
        refined_camera = apply_image_similarity(
            rotation, ppw, tx, ty, values, affine_refine
        )
        refined_param = build_param(*refined_camera, old_param)
        refined_param_path = out_dir / "front_dense_silhouette.npy"
        np.save(refined_param_path, refined_param)
        cv2.imwrite(str(out_dir / "hair_silhouette_before.png"), source_mask)
        cv2.imwrite(
            str(out_dir / "hair_silhouette_refined.png"),
            refined_mask.astype(np.uint8) * 255,
        )
        silhouette_report = {
            "render_mask": str(render_mask_path),
            "target_mask": str(target_mask_path),
            "selected_hair_faces": int(hair_faces.sum()),
            "iou_before": float(iou_before),
            "iou_after": float(iou_after),
            "image_scale": float(np.exp(values[0])),
            "image_angle_deg": float(values[1]),
            "image_translation_px": values[2:4].astype(float).tolist(),
            "refined_euler_xyz_deg": Rotation.from_matrix(
                refined_camera[0]
            ).as_euler("xyz", degrees=True).tolist(),
            "refined_scale": float(refined_param["scale"][0]),
            "refined_center": refined_param["center"].reshape(-1).astype(float).tolist(),
        }
    elif args.require_silhouette:
        missing = []
        if not render_mask_path.exists():
            missing.append(str(render_mask_path))
        if not target_mask_path.exists():
            missing.append(str(target_mask_path))
        raise RuntimeError("无法完成头发轮廓标定，缺少: " + ", ".join(missing))

    matches = draw_matches(
        render_image, target_image, render_xy_512, target_xy, scores
    )
    reprojection = draw_reprojection(target_image, target_xy, predicted, errors)
    cv2.imwrite(str(out_dir / "dino_matches.png"), matches)
    cv2.imwrite(str(out_dir / "dense_reprojection.png"), reprojection)

    calib = load_calib(str(param_path), loadSize=1024).numpy()
    report = {
        "model": MODEL_NAME,
        "device": device,
        "mutual_matches": mutual_count,
        "ransac_matches": int(len(valid)),
        "surface_matches": int(valid.sum()),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.quantile(errors, 0.90)),
        "dino_score_mean": float(scores.mean()),
        "render_to_target_affine": affine.tolist(),
        "euler_xyz_deg": Rotation.from_matrix(rotation).as_euler(
            "xyz", degrees=True
        ).tolist(),
        "det_R": float(np.linalg.det(rotation)),
        "orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
        ),
        "pixels_per_world": float(ppw),
        "scale": float(param["scale"][0]),
        "center": param["center"].reshape(-1).astype(float).tolist(),
        "pixel_translation": [float(tx), float(ty)],
        "calib": calib.astype(float).tolist(),
        "silhouette_refinement": silhouette_report,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"param={param_path}")
    if refined_param_path is not None:
        print(f"refined_param={refined_param_path}")
    print(f"matches={out_dir / 'dino_matches.png'}")
    print(f"reprojection={out_dir / 'dense_reprojection.png'}")


if __name__ == "__main__":
    main()
