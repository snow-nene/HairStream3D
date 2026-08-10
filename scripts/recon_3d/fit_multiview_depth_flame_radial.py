"""以完整光头表面为内层、Pixal3D 对齐 mesh 为外层提取径向头发体素。"""

import argparse
import json
import os
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, binary_dilation
from skimage.measure import marching_cubes


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.recon_3d.fit_multiview_depth_visual_hull import (
    compact_visible_lines,
    raycast_curve_points,
)
from scripts.recon_3d.fit_multiview_depth_with_flame import (
    load_calibrations,
    load_lines,
    project,
    save_lines,
)


def load_obj_vertices(path):
    points = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
    if not points:
        raise ValueError(f"OBJ 中没有顶点: {path}")
    return np.asarray(points, dtype=np.float64)


def load_seg_masks(data_dir, views):
    masks = {}
    for view in views:
        image = imageio.imread(os.path.join(data_dir, "maps", "seg", f"{view}.png"))
        if image.ndim == 3:
            image = image[..., 0]
        masks[view] = image > 127
    return masks


def points_in_any_mask(points, masks, calibs):
    selected = np.zeros(len(points), dtype=bool)
    for view, mask in masks.items():
        uv, _ = project(points, calibs[view])
        height, width = mask.shape
        px = np.rint((uv[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        py = np.rint((uv[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        ids = np.flatnonzero(inside)
        selected[ids] |= mask[py[ids], px[ids]]
    return selected


def sample_bald_head_surface(head_mesh, sample_count, seed):
    """密集采样整个闭合光头表面，而不是仅采样 roots 所在头皮区域。"""
    head_mesh.compute_vertex_normals()
    o3d.utility.random.seed(seed)
    samples = head_mesh.sample_points_uniformly(
        number_of_points=sample_count, use_triangle_normal=True
    )
    points = np.asarray(samples.points).astype(np.float64)
    normals = np.asarray(samples.normals).astype(np.float64)
    center = head_mesh.get_axis_aligned_bounding_box().get_center()
    flip = np.einsum("ij,ij->i", normals, points - center) < 0.0
    normals[flip] *= -1.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return points, normals


def raycast_surface_to_outer(
    surface_points,
    normals,
    outer_scene,
    head_scene,
    max_thickness,
    passes,
):
    """从完整光头表面沿外法线寻找 Pixal3D mesh 的首个外部交点。"""
    epsilon = 2e-4
    origins = surface_points + epsilon * normals
    travelled = np.full(len(surface_points), epsilon, dtype=np.float64)
    active = np.arange(len(surface_points))
    outer_points = np.full_like(surface_points, np.nan)
    thickness = np.full(len(surface_points), np.nan, dtype=np.float64)

    for _ in range(passes):
        if not len(active):
            break
        rays = np.column_stack([origins[active], normals[active]]).astype(np.float32)
        hit = outer_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        finite = np.isfinite(hit)
        ids = active[finite]
        if not len(ids):
            break
        candidate = origins[ids] + normals[ids] * hit[finite, None]
        total = travelled[ids] + hit[finite]
        head_sdf = head_scene.compute_signed_distance(
            o3d.core.Tensor(candidate.astype(np.float32))
        ).numpy()
        accept = (
            (total <= max_thickness)
            & (total >= 1e-3)
            & (head_sdf >= -5e-4)
        )
        accepted = ids[accept]
        outer_points[accepted] = candidate[accept]
        thickness[accepted] = total[accept]

        rejected = ids[~accept]
        rejected_hit = hit[finite][~accept]
        origins[rejected] += normals[rejected] * (rejected_hit[:, None] + epsilon)
        travelled[rejected] += rejected_hit + epsilon
        active = rejected[travelled[rejected] < max_thickness]
    return outer_points, thickness


def voxelize_radial_volume(surface_points, outer_points, valid, resolution):
    """把所有有效的内外表面线段扫掠成规则占据体素。"""
    inner = surface_points[valid]
    outer = outer_points[valid]
    if not len(inner):
        raise RuntimeError("完整光头表面没有射中有效外壳")
    lower = np.minimum(inner.min(axis=0), outer.min(axis=0))
    upper = np.maximum(inner.max(axis=0), outer.max(axis=0))
    voxel_size = float((upper - lower).max() / max(resolution - 5, 1))
    origin = lower - 2.0 * voxel_size
    shape = np.ceil((upper - origin + 2.0 * voxel_size) / voxel_size).astype(int) + 1
    occupancy = np.zeros(tuple(shape), dtype=bool)

    direction = outer - inner
    lengths = np.linalg.norm(direction, axis=1)
    for start in range(0, len(inner), 4096):
        stop = min(start + 4096, len(inner))
        chunks = []
        for point, delta, length in zip(
            inner[start:stop], direction[start:stop], lengths[start:stop]
        ):
            count = max(2, int(np.ceil(length / (0.5 * voxel_size))) + 1)
            chunks.append(point + np.linspace(0.0, 1.0, count)[:, None] * delta)
        samples = np.vstack(chunks)
        indices = np.rint((samples - origin) / voxel_size).astype(np.int32)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        indices = indices[inside]
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    occupancy = binary_dilation(occupancy, iterations=1)
    occupancy = binary_closing(occupancy, iterations=2)
    return occupancy, origin, voxel_size


def occupancy_to_mesh(occupancy, origin, voxel_size):
    vertices, faces, _, _ = marching_cubes(
        occupancy.astype(np.float32), level=0.5, spacing=(voxel_size,) * 3
    )
    vertices += origin
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def save_surface_correspondences(path, surface_points, outer_points, valid):
    points = np.vstack([surface_points[valid], outer_points[valid]])
    count = int(valid.sum())
    lines = np.column_stack([np.arange(count), np.arange(count) + count])
    save_lines(path, points, lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_id", required=True)
    parser.add_argument("--views", nargs="+", default=["front", "left"])
    parser.add_argument("--head_mesh", default="data/head_model.obj")
    parser.add_argument("--roots", default="data/roots10k.obj")
    parser.add_argument("--target_mesh", default=None)
    parser.add_argument("--max_thickness", type=float, default=0.08)
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--surface_samples", type=int, default=60000)
    parser.add_argument("--voxel_resolution", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join("results", "multiview_data", args.img_id)
    target_path = args.target_mesh or os.path.join(
        data_dir, "pixal3d", "hair_mesh_aligned_best.obj"
    )
    input_dir = os.path.join(data_dir, "pde_reconstruction", "per_view")
    output_dir = os.path.join(
        data_dir, "pde_reconstruction", "depth_fit_experiment_flame_radial"
    )
    os.makedirs(output_dir, exist_ok=True)
    calibs = load_calibrations(data_dir, args.views)
    masks = load_seg_masks(data_dir, args.views)

    head_mesh = o3d.io.read_triangle_mesh(args.head_mesh)
    outer_mesh = o3d.io.read_triangle_mesh(target_path)
    roots = load_obj_vertices(args.roots)
    head_scene = o3d.t.geometry.RaycastingScene()
    head_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head_mesh))
    outer_scene = o3d.t.geometry.RaycastingScene()
    outer_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(outer_mesh))
    surface_points, normals = sample_bald_head_surface(
        head_mesh, args.surface_samples, args.seed
    )
    outer_points, thickness = raycast_surface_to_outer(
        surface_points,
        normals,
        outer_scene,
        head_scene,
        args.max_thickness,
        args.passes,
    )
    valid = np.isfinite(thickness)
    occupancy, voxel_origin, voxel_size = voxelize_radial_volume(
        surface_points, outer_points, valid, args.voxel_resolution
    )
    volume_mesh = occupancy_to_mesh(occupancy, voxel_origin, voxel_size)

    # 外壳必须保持为 Pixal3D 对齐 mesh；不再用 FLAME 拓扑伪造外壳。
    outer_path = os.path.join(output_dir, "radial_hair_shell.obj")
    shutil.copy2(target_path, outer_path)
    o3d.io.write_point_cloud(
        os.path.join(output_dir, "bald_head_samples.ply"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surface_points)),
    )
    save_surface_correspondences(
        os.path.join(output_dir, "head_to_outer_rays.ply"),
        surface_points,
        outer_points,
        valid,
    )
    np.savez_compressed(
        os.path.join(output_dir, "radial_hair_volume.npz"),
        occupancy=occupancy,
        origin=voxel_origin,
        voxel_size=np.asarray(voxel_size),
    )
    o3d.io.write_triangle_mesh(
        os.path.join(output_dir, "radial_hair_volume.obj"), volume_mesh
    )

    metrics = {
        "views": args.views,
        "outer_shell": target_path,
        "inner_head_mesh": args.head_mesh,
        "roots_for_later_strand_growth": args.roots,
        "root_count": int(len(roots)),
        "surface_sample_count": int(len(surface_points)),
        "valid_surface_hits": int(valid.sum()),
        "valid_surface_hit_fraction": float(valid.mean()),
        "voxel_resolution_limit": args.voxel_resolution,
        "voxel_shape": list(occupancy.shape),
        "voxel_size_m": voxel_size,
        "occupied_voxels": int(occupancy.sum()),
        "thickness_quantiles_m": (
            np.quantile(thickness[valid], [0.0, 0.1, 0.5, 0.9, 1.0]).tolist()
            if valid.any()
            else []
        ),
        "per_view": {},
    }
    # 每个视图只把本视图曲线投到真实外壳，不再投到变形 FLAME。
    for view in args.views:
        points, lines = load_lines(os.path.join(input_dir, view, "hair.ply"))
        hit_points, hit_valid = raycast_curve_points(points, calibs[view], outer_scene)
        # seg 只在视图投影阶段过滤；不能反向裁剪完整的 3D 候选体。
        hit_valid &= points_in_any_mask(
            hit_points, {view: masks[view]}, {view: calibs[view]}
        )
        output_points, output_lines = compact_visible_lines(hit_points, lines, hit_valid)
        save_lines(os.path.join(output_dir, view, "hair.ply"), output_points, output_lines)
        metrics["per_view"][view] = {
            "source_points": int(len(points)),
            "output_points": int(len(output_points)),
            "ray_hit_fraction": float(hit_valid.mean()),
        }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
