"""用多视图头发轮廓和 90% FLAME 构建公制 visual hull 深度。"""

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.recon_3d.fit_multiview_depth_with_flame import (
    load_calibrations,
    load_lines,
    make_scaled_flame,
    project,
    save_lines,
    trimmed_mean,
    unproject,
)


def carve_visual_hull(masks, calibs, bounds_min, bounds_max, resolution, inner_scene):
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution, dtype=np.float32)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)

    slab_size = 8
    for x_start in range(0, resolution, slab_size):
        x_stop = min(x_start + slab_size, resolution)
        gx, gy, gz = np.meshgrid(
            xs[x_start:x_stop], ys, zs, indexing="ij"
        )
        points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        keep = np.ones(len(points), dtype=bool)
        for view, mask in masks.items():
            uv, _ = project(points, calibs[view])
            height, width = mask.shape
            px = np.rint((uv[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
            py = np.rint((uv[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            selected = np.zeros(len(points), dtype=bool)
            ids = np.where(inside)[0]
            selected[ids] = mask[py[ids], px[ids]]
            keep &= selected
        candidate_ids = np.where(keep)[0]
        if len(candidate_ids):
            signed_distance = inner_scene.compute_signed_distance(
                o3d.core.Tensor(points[candidate_ids].astype(np.float32))
            ).numpy()
            keep[candidate_ids[signed_distance < 0.0]] = False
        occupancy[x_start:x_stop] = keep.reshape(
            x_stop - x_start, resolution, resolution
        )
        print(
            f"  carve x={x_stop}/{resolution}, occupied={occupancy[:x_stop].sum()}"
        )
    return occupancy, xs, ys, zs


def occupancy_to_mesh(occupancy, xs, ys, zs):
    spacing = (
        float(xs[1] - xs[0]),
        float(ys[1] - ys[0]),
        float(zs[1] - zs[0]),
    )
    vertices, faces, _, _ = marching_cubes(
        occupancy.astype(np.float32), level=0.5, spacing=spacing
    )
    vertices += np.array([xs[0], ys[0], zs[0]], dtype=np.float32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def raycast_curve_points(points, calib, hull_scene):
    uv, _ = project(points, calib)
    near = unproject(uv, np.full(len(uv), 1.5), calib)
    far = unproject(uv, np.full(len(uv), -0.5), calib)
    direction = far - near
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    rays = np.column_stack([near, direction]).astype(np.float32)
    hit = hull_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    valid = np.isfinite(hit)
    hit_points = np.zeros_like(points)
    hit_points[valid] = near[valid] + direction[valid] * hit[valid, None]
    return hit_points, valid


def compact_visible_lines(points, lines, valid):
    mapping = np.full(len(points), -1, dtype=np.int32)
    mapping[valid] = np.arange(valid.sum(), dtype=np.int32)
    kept_lines = valid[lines[:, 0]] & valid[lines[:, 1]]
    return points[valid], mapping[lines[kept_lines]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--views", nargs="+", default=["front", "left"])
    parser.add_argument("--flame", default="data/head_model.obj")
    parser.add_argument("--flame_scale", type=float, default=0.9)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--padding", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = os.path.join("results", "multiview_data", args.img_id)
    input_dir = os.path.join(data_dir, "pde_reconstruction", "per_view")
    output_dir = os.path.join(
        data_dir, "pde_reconstruction", "depth_fit_experiment_visual_hull"
    )
    os.makedirs(output_dir, exist_ok=True)
    calibs = load_calibrations(data_dir, args.views)
    masks = {
        view: imageio.imread(
            os.path.join(data_dir, "maps", "strand_map", f"{view}.png")
        )[:, :, 0] > 25
        for view in args.views
    }

    head_mesh = o3d.io.read_triangle_mesh(args.flame)
    head_bounds = head_mesh.get_axis_aligned_bounding_box()
    bounds_min = head_bounds.get_min_bound() - args.padding
    bounds_max = head_bounds.get_max_bound() + args.padding
    inner_mesh = make_scaled_flame(args.flame, args.flame_scale)
    inner_scene = o3d.t.geometry.RaycastingScene()
    inner_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(inner_mesh))

    occupancy, xs, ys, zs = carve_visual_hull(
        masks,
        calibs,
        bounds_min,
        bounds_max,
        args.resolution,
        inner_scene,
    )
    hull_mesh = occupancy_to_mesh(occupancy, xs, ys, zs)
    hull_path = os.path.join(output_dir, "visual_hull.obj")
    o3d.io.write_triangle_mesh(hull_path, hull_mesh)
    hull_scene = o3d.t.geometry.RaycastingScene()
    hull_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull_mesh))
    head_scene = o3d.t.geometry.RaycastingScene()
    head_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head_mesh))

    metrics = {
        "views": args.views,
        "flame_scale": args.flame_scale,
        "resolution": args.resolution,
        "voxel_size_m": [
            float(xs[1] - xs[0]),
            float(ys[1] - ys[0]),
            float(zs[1] - zs[0]),
        ],
        "occupied_voxels": int(occupancy.sum()),
        "hull_vertices": int(len(hull_mesh.vertices)),
        "hull_triangles": int(len(hull_mesh.triangles)),
        "per_view": {},
    }
    view_points = {}
    for view in args.views:
        source_points, source_lines = load_lines(
            os.path.join(input_dir, view, "hair.ply")
        )
        hit_points, valid = raycast_curve_points(
            source_points, calibs[view], hull_scene
        )
        points, lines = compact_visible_lines(hit_points, source_lines, valid)
        view_dir = os.path.join(output_dir, view)
        save_lines(os.path.join(view_dir, "hair.ply"), points, lines)
        view_points[view] = points
        head_sdf = head_scene.compute_signed_distance(
            o3d.core.Tensor(points.astype(np.float32))
        ).numpy()
        reprojection_uv, _ = project(points, calibs[view])
        height, width = masks[view].shape
        px = np.rint(
            (reprojection_uv[:, 0] + 1.0) * 0.5 * (width - 1)
        ).astype(np.int32)
        py = np.rint(
            (reprojection_uv[:, 1] + 1.0) * 0.5 * (height - 1)
        ).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        in_mask = np.zeros(len(points), dtype=bool)
        ids = np.where(inside)[0]
        in_mask[ids] = masks[view][py[ids], px[ids]]
        metrics["per_view"][view] = {
            "source_points": int(len(source_points)),
            "hit_points": int(len(points)),
            "ray_hit_fraction": float(valid.mean()),
            "own_mask_fraction": float(in_mask.mean()),
            "inside_original_head_fraction": float((head_sdf < 0.0).mean()),
            "inside_original_head_2mm_fraction": float((head_sdf < -0.002).mean()),
            "head_sdf_quantiles_m": np.quantile(
                head_sdf, [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist(),
        }

    if len(args.views) == 2:
        first, second = args.views
        p0, p1 = view_points[first], view_points[second]
        metrics["symmetric_trimmed_chamfer_m"] = trimmed_mean(
            cKDTree(p1).query(p0, workers=-1)[0]
        ) + trimmed_mean(cKDTree(p0).query(p1, workers=-1)[0])

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
