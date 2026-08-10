"""用 90% FLAME 内层锚点拟合多视图相对头发深度。"""

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.recon_3d.recon3D import load_calib


def load_side_calibration(data_dir, view):
    camera = np.load(
        os.path.join(data_dir, "blender_renders", "camera_params", f"{view}.npz")
    )
    transform = np.load(os.path.join(data_dir, "pixal3d", "glb_to_world.npz"))
    umeyama = np.eye(4, dtype=np.float64)
    umeyama[:3, :3] = (
        float(np.asarray(transform["umeyama_scale"]).reshape(-1)[0])
        * transform["umeyama_R"]
    )
    umeyama[:3, 3] = transform["umeyama_t"]
    gltf_to_hairstep = transform["icp_T"] @ umeyama
    gltf_to_blender = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    hairstep_to_blender = gltf_to_blender @ np.linalg.inv(gltf_to_hairstep)
    calib = (
        np.diag([1.0, -1.0, 1.0, 1.0])
        @ camera["world_to_clip"]
        @ hairstep_to_blender
    )
    hair_to_camera = camera["world_to_camera"] @ hairstep_to_blender
    depth_min = float(camera["camera_depth_min"])
    depth_max = float(camera["camera_depth_max"])
    calib[2] = (hair_to_camera[2] - depth_min * hair_to_camera[3]) / max(
        depth_max - depth_min, 1e-8
    )
    return calib.astype(np.float64)


def load_calibrations(data_dir, views):
    calibrations = {}
    for view in views:
        if view == "front":
            calib = load_calib(
                os.path.join(data_dir, "maps", "param", "front.npy"), loadSize=1024
            )
            if isinstance(calib, torch.Tensor):
                calib = calib.numpy()
            calibrations[view] = np.asarray(calib, dtype=np.float64)
        else:
            calibrations[view] = load_side_calibration(data_dir, view)
    return calibrations


def load_lines(path):
    line_set = o3d.io.read_line_set(path)
    return np.asarray(line_set.points), np.asarray(line_set.lines)


def save_lines(path, points, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    if not o3d.io.write_line_set(path, line_set):
        raise RuntimeError(f"无法写入 {path}")


def project(points, calib):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ calib.T
    uv = projected[:, :2] / projected[:, 3:4]
    depth = projected[:, 2] / projected[:, 3]
    return uv, depth


def unproject(uv, depth, calib):
    clip = np.column_stack([uv, depth, np.ones(len(uv))])
    world_h = clip @ np.linalg.inv(calib).T
    return world_h[:, :3] / world_h[:, 3:4]


def make_scaled_flame(path, scale):
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    center = mesh.get_axis_aligned_bounding_box().get_center()
    scaled = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(center + scale * (vertices - center)),
        mesh.triangles,
    )
    return scaled


def scalp_depth_for_uv(uv, calib, scene):
    near = unproject(uv, np.full(len(uv), 1.5), calib)
    far = unproject(uv, np.full(len(uv), -0.5), calib)
    direction = far - near
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    rays = np.column_stack([near, direction]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    valid = np.isfinite(hit)
    hit_points = np.zeros_like(near)
    hit_points[valid] = near[valid] + direction[valid] * hit[valid, None]
    depth = np.full(len(uv), np.nan)
    if valid.any():
        _, depth[valid] = project(hit_points[valid], calib)
    return depth


def trimmed_mean(values, keep=0.8):
    if len(values) == 0:
        return 0.0
    count = max(1, int(len(values) * keep))
    return float(np.partition(values, count - 1)[:count].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--views", nargs=2, default=["front", "left"])
    parser.add_argument("--flame", default="data/head_model.obj")
    parser.add_argument("--flame_scale", type=float, default=0.9)
    parser.add_argument("--sample_points", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join("results", "multiview_data", args.img_id)
    input_dir = os.path.join(data_dir, "pde_reconstruction", "per_view")
    output_dir = os.path.join(
        data_dir, "pde_reconstruction", "depth_fit_experiment_shell"
    )
    views = args.views
    calibs = load_calibrations(data_dir, views)

    scaled_flame = make_scaled_flame(args.flame, args.flame_scale)
    flame_scene = o3d.t.geometry.RaycastingScene()
    flame_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scaled_flame))
    head_scene = o3d.t.geometry.RaycastingScene()
    head_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(args.flame)
        )
    )

    rng = np.random.default_rng(args.seed)
    records = {}
    for view in views:
        points, lines = load_lines(os.path.join(input_dir, view, "hair.ply"))
        uv, relative_depth = project(points, calibs[view])
        count = min(args.sample_points, len(points))
        sample_ids = rng.choice(len(points), count, replace=False)
        sample_uv = uv[sample_ids]
        sample_depth = relative_depth[sample_ids]
        inner_depth = scalp_depth_for_uv(uv, calibs[view], flame_scene)
        outer_depth = scalp_depth_for_uv(uv, calibs[view], head_scene)

        def fill_missing(depth):
            valid = np.isfinite(depth)
            if not valid.any():
                raise RuntimeError(f"{view} 没有任何 FLAME 射线交点")
            filled = depth.copy()
            missing = ~valid
            if missing.any():
                nearest = cKDTree(uv[valid]).query(uv[missing], workers=-1)[1]
                filled[missing] = depth[valid][nearest]
            return filled

        inner_depth = fill_missing(inner_depth)
        outer_depth = fill_missing(outer_depth)
        shell_depth = np.maximum(outer_depth - inner_depth, 1e-4)
        mask = imageio.imread(
            os.path.join(data_dir, "maps", "strand_map", f"{view}.png")
        )[:, :, 0] > 25
        outside_distance = distance_transform_edt(~mask).astype(np.float64)
        records[view] = {
            "points": points,
            "lines": lines,
            "uv": uv,
            "relative_depth": relative_depth,
            "sample_uv": sample_uv,
            "sample_depth": sample_depth,
            "outer_depth": outer_depth,
            "shell_depth": shell_depth,
            "sample_outer_depth": outer_depth[sample_ids],
            "sample_shell_depth": shell_depth[sample_ids],
            "mask": mask,
            "outside_distance": outside_distance,
        }

    def transformed_samples(params, view_index):
        view = views[view_index]
        thickness_scale = params[view_index]
        record = records[view]
        depth = (
            record["sample_outer_depth"]
            + thickness_scale
            * record["sample_shell_depth"]
            * record["sample_depth"]
        )
        return unproject(record["sample_uv"], depth, calibs[view]), depth

    def silhouette_loss(points, target_view):
        record = records[target_view]
        uv, _ = project(points, calibs[target_view])
        height, width = record["mask"].shape
        px = np.rint((uv[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        py = np.rint((uv[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        outside = (px < 0) | (px >= width) | (py < 0) | (py >= height)
        distances = np.full(len(points), 32.0)
        inside = ~outside
        distances[inside] = record["outside_distance"][py[inside], px[inside]]
        return trimmed_mean(distances / max(height, width), keep=0.7)

    def objective(params):
        p0, d0 = transformed_samples(params, 0)
        p1, d1 = transformed_samples(params, 1)
        dist01 = cKDTree(p1).query(p0, workers=-1)[0]
        dist10 = cKDTree(p0).query(p1, workers=-1)[0]
        chamfer = trimmed_mean(dist01) + trimmed_mean(dist10)

        silhouette = silhouette_loss(p0, views[1]) + silhouette_loss(p1, views[0])
        return chamfer + 0.15 * silhouette

    bounds = [(0.0, 12.0)] * len(views)
    global_result = differential_evolution(
        objective,
        bounds,
        seed=args.seed,
        popsize=8,
        maxiter=18,
        polish=False,
        workers=1,
        updating="immediate",
    )
    result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 80, "xtol": 1e-5, "ftol": 1e-6},
    )

    metrics = {
        "views": views,
        "flame_scale": args.flame_scale,
        "objective": float(result.fun),
        "success": bool(result.success),
        "parameters": {},
    }
    transformed = {}
    for view_index, view in enumerate(views):
        thickness_scale = result.x[view_index]
        record = records[view]
        fitted_depth = (
            record["outer_depth"]
            + thickness_scale
            * record["shell_depth"]
            * record["relative_depth"]
        )
        fitted_points = unproject(record["uv"], fitted_depth, calibs[view])
        transformed[view] = fitted_points
        sdf = head_scene.compute_signed_distance(
            o3d.core.Tensor(fitted_points.astype(np.float32))
        ).numpy()
        view_dir = os.path.join(output_dir, view)
        save_lines(
            os.path.join(view_dir, "hair.ply"), fitted_points, record["lines"]
        )
        metrics["parameters"][view] = {
            "thickness_scale": float(thickness_scale),
            "points": int(len(fitted_points)),
            "inside_original_head_fraction": float((sdf < 0.0).mean()),
            "inside_original_head_2mm_fraction": float((sdf < -0.002).mean()),
            "head_sdf_quantiles_m": np.quantile(
                sdf, [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist(),
        }

    p0, p1 = transformed[views[0]], transformed[views[1]]
    metrics["symmetric_trimmed_chamfer_m"] = trimmed_mean(
        cKDTree(p1).query(p0, workers=-1)[0]
    ) + trimmed_mean(cKDTree(p0).query(p1, workers=-1)[0])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
