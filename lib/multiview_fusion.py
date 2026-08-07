"""
Multi-View Strand & Depth Fusion for 3D Hair Reconstruction.

Builds a 3D orientation volume by fusing 2D strand maps from 4 orthographic
camera views (front, left, right, back).

Hard constraints:
  1. Front view is ABSOLUTE GROUND TRUTH — where the front camera can see,
     no other view contributes.  The front strand direction is used as-is.
  2. Direction consistency — at view boundaries, transition weights ensure
     smooth blending with no abrupt direction flips.

Algorithm:
  For each voxel, project to all 4 cameras:
    - If front sees it → use front direction exclusively.
    - Else, weighted blend of left/right/back directions.
    - Unseen voxels → left as zeros (PDE fills them via Laplace interpolation).
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def align_vector_sign(vectors, reference):
    """Flip 180-degree-ambiguous vectors into the reference hemisphere."""
    flip = torch.sum(vectors * reference, dim=0, keepdim=True) < 0.0
    return torch.where(flip, -vectors, vectors)


def limit_direction_normal_component(directions, normals, max_component=0.3):
    """Clamp excessive surface-normal motion while preserving strand tangent."""
    directions = np.asarray(directions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    result = directions.copy()
    normal_component = np.sum(directions * normals, axis=1)
    tangent = directions - normal_component[:, None] * normals
    tangent_norm = np.linalg.norm(tangent, axis=1)
    needs_clamp = (
        (np.abs(normal_component) > float(max_component))
        & (tangent_norm > 1e-6)
    )
    if needs_clamp.any():
        target_normal = np.clip(
            normal_component[needs_clamp],
            -float(max_component),
            float(max_component),
        )
        tangent_unit = tangent[needs_clamp] / tangent_norm[needs_clamp, None]
        tangent_weight = np.sqrt(np.maximum(1.0 - target_normal**2, 0.0))
        result[needs_clamp] = (
            tangent_weight[:, None] * tangent_unit
            + target_normal[:, None] * normals[needs_clamp]
        )
    return result, needs_clamp


def orient_sparse_direction_axes(
    flat_ids,
    axes,
    reference_directions,
    ownership,
    volume_shape,
    neighbor_radius=3.5,
    preferred_owners=None,
    preferred_direction=(0.0, -1.0, 0.0),
    preferred_min_alignment=0.15,
):
    """Give sparse signless direction axes a spatially consistent sign.

    Eigenvectors of ``v v^T`` have an arbitrary sign.  Propagating signs over
    nearby samples preserves the smooth image-space strand axis, while the
    original lifted directions choose one global sign per connected component.
    Ownership prevents propagation across unrelated camera-view boundaries.
    Selected side-view owners may use gravity to resolve the component-wide
    sign; nearly horizontal components retain their lifted-view reference.
    """
    from collections import deque
    from scipy.spatial import cKDTree

    flat_ids = np.asarray(flat_ids, dtype=np.int64)
    result = np.asarray(axes, dtype=np.float32).copy()
    references = np.asarray(reference_directions, dtype=np.float32)
    ownership = np.asarray(ownership)
    preferred_owners = set(
        np.asarray(preferred_owners if preferred_owners is not None else []).tolist()
    )
    preferred_direction = np.asarray(preferred_direction, dtype=np.float32)
    preferred_direction /= np.linalg.norm(preferred_direction) + 1e-8
    if len(flat_ids) == 0:
        return result

    coordinates = np.column_stack(
        np.unravel_index(flat_ids, tuple(volume_shape))
    ).astype(np.float32)
    for owner in np.unique(ownership):
        owner_ids = np.flatnonzero(ownership == owner)
        if len(owner_ids) == 0:
            continue
        owner_coords = coordinates[owner_ids]
        tree = cKDTree(owner_coords)
        neighbors = tree.query_ball_point(owner_coords, r=float(neighbor_radius))
        visited = np.zeros(len(owner_ids), dtype=bool)

        for seed in range(len(owner_ids)):
            if visited[seed]:
                continue
            visited[seed] = True
            component = [seed]
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                current_axis = result[owner_ids[current]]
                for neighbor in neighbors[current]:
                    if neighbor == current or visited[neighbor]:
                        continue
                    neighbor_id = owner_ids[neighbor]
                    if np.dot(result[neighbor_id], current_axis) < 0.0:
                        result[neighbor_id] *= -1.0
                    visited[neighbor] = True
                    component.append(neighbor)
                    queue.append(neighbor)

            # The traversal tree fixes parent-child signs.  A few conflicting
            # edges can remain around loops, so optimize the local agreement
            # objective with deterministic coordinate-descent sweeps.
            for _ in range(8):
                changed = False
                for current in component:
                    adjacent = [
                        neighbor for neighbor in neighbors[current]
                        if neighbor != current and visited[neighbor]
                    ]
                    if not adjacent:
                        continue
                    neighbor_ids = owner_ids[np.asarray(adjacent, dtype=np.int64)]
                    neighbor_sum = result[neighbor_ids].sum(axis=0)
                    current_id = owner_ids[current]
                    if np.dot(result[current_id], neighbor_sum) < 0.0:
                        result[current_id] *= -1.0
                        changed = True
                if not changed:
                    break

            component_ids = owner_ids[np.asarray(component, dtype=np.int64)]
            use_preferred = False
            if owner in preferred_owners:
                preferred_alignment = (
                    result[component_ids] @ preferred_direction
                )
                use_preferred = (
                    np.mean(np.abs(preferred_alignment))
                    >= float(preferred_min_alignment)
                )
            if use_preferred:
                anchor_score = np.sum(preferred_alignment)
            else:
                anchor_score = np.sum(
                    result[component_ids] * references[component_ids]
                )
            if anchor_score < 0.0:
                result[component_ids] *= -1.0

    return result


def build_blender_calib(view, center, extent):
    """Build a 4×4 orthographic calibration matrix for a camera view.
    Matches the exact projection logic of load_calib in recon3D.py,
    where Y is up, +Z is front (face), and Y is inverted in projection.
    """
    scale = 1.0
    # The original model scale factor is roughly 256.0 / (extent * 1.4)
    # But wait, loadSize/2 = 512 (for loadSize=1024), 
    # To map world extent (0.3) to NDC [-1, 1], the scale factor is 1.0 / (extent * 1.4)
    ortho_ratio = extent * 1.4

    # Rotations for the 4 views. Front is Identity.
    # We rotate the WORLD into CAMERA coordinates.
    # Since front is Identity, Camera Z = World Z.
    # For back, we look from the back (-Z), so we rotate 180 degrees around Y.
    import math
    if view == "front":
        angles = [0.0, 0.0, 0.0]
    elif view == "back":
        angles = [0.0, math.pi, 0.0]
    elif view == "left":
        # Look from the left (+X). Rotate world by -90 deg around Y.
        angles = [0.0, -math.pi / 2, 0.0]
    elif view == "right":
        # Look from the right (-X). Rotate world by 90 deg around Y.
        angles = [0.0, math.pi / 2, 0.0]
    else:
        angles = [0.0, 0.0, 0.0]

    import scipy.spatial.transform as sst
    R = sst.Rotation.from_euler('xyz', angles).as_matrix().astype(np.float32)

    translate = -np.matmul(R, center.reshape(3, 1))
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]], dtype=np.float32)], 0)

    # intrinsic
    scale_intrinsic = np.identity(4, dtype=np.float32)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio  # INVERT Y!
    scale_intrinsic[2, 2] = scale / ortho_ratio
    
    calib_mat = np.matmul(scale_intrinsic, extrinsic)
    
    return calib_mat, R




def decode_strand_2d(strand_map, px, py):
    """Decode 2D strand direction from strand_map at given pixel coords.

    Strand_map encoding (from HairStep dataset):
      R = 255 for hair
      G = (dy_img + 1) / 2 * 255     dy_img > 0 = down in image
      B = (-dx_img + 1) / 2 * 255    dx_img > 0 = right in image

    BUT: the camera Y axis points UP (world Z), while dy_img positive
    means DOWN (toward larger pixel rows).  We must NEGATE dy to get
    the correct camera-space Y component.

    Returns:
        dx: (N,) float32, image-X component (right = positive)
        dy: (N,) float32, camera-Y component (up = positive)  ← NEGATED from image
        mask: (N,) bool
    """
    H, W = strand_map.shape[:2]
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    r = np.zeros_like(px, dtype=np.float32)
    g = np.zeros_like(px, dtype=np.float32)
    b = np.zeros_like(px, dtype=np.float32)

    r[valid] = strand_map[py[valid], px[valid], 0]  # mask
    g[valid] = strand_map[py[valid], px[valid], 1]  # (dy_img + 1)/2
    b[valid] = strand_map[py[valid], px[valid], 2]  # (-dx_img + 1)/2

    # G = (dy_img + 1)/2 → dy_img = 2*G - 1  (positive = down in image)
    # Camera Y = UP in image, so camera_Y_component = -dy_img = -(2*G - 1)
    dy = -(2.0 * g - 1.0)  # NEGATED: positive = UP in camera Y
    dx = 1.0 - 2.0 * b     # positive = RIGHT in camera X

    mask = (r > 0.1) & valid

    return dx, dy, mask


def trace_view_strands_3d(
    strand_map,
    depth_map,
    calib,
    seed_spacing=4,
    step_pixels=1.5,
    max_steps=512,
    min_curve_points=6,
):
    """Trace one view's observed 2D orientation field and lift it to 3D.

    The strand map is an undirected line field, so each seed is traced in both
    directions and each new sample is sign-aligned with the previous tangent.
    Every returned point comes directly from this view's depth map; no fused or
    front-baseline orientation is consulted.
    """
    strand_map = np.asarray(strand_map, dtype=np.float32)
    depth_map = np.asarray(depth_map, dtype=np.float32)
    height, width = depth_map.shape
    if strand_map.shape[:2] != depth_map.shape:
        raise ValueError(
            f"strand/depth shape mismatch: {strand_map.shape[:2]} vs "
            f"{depth_map.shape}"
        )

    hair_mask = (strand_map[:, :, 0] > 0.1) & (depth_map > 0.05)
    direction = np.stack(
        [1.0 - 2.0 * strand_map[:, :, 2],
         2.0 * strand_map[:, :, 1] - 1.0],
        axis=-1,
    )
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8
    direction[~hair_mask] = 0.0

    def sample(array, point):
        x, y = point
        if x < 0 or x > width - 1 or y < 0 or y > height - 1:
            return None
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        wx, wy = x - x0, y - y0
        return (
            array[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + array[y0, x1] * wx * (1.0 - wy)
            + array[y1, x0] * (1.0 - wx) * wy
            + array[y1, x1] * wx * wy
        )

    def valid(point):
        x, y = np.rint(point).astype(np.int32)
        return 0 <= x < width and 0 <= y < height and hair_mask[y, x]

    def follow(seed, sign):
        point = np.asarray(seed, dtype=np.float32)
        tangent = sample(direction, point)
        if tangent is None or np.linalg.norm(tangent) < 1e-4:
            return [point]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8) * sign
        curve = [point.copy()]
        for _ in range(max_steps):
            midpoint = point + 0.5 * step_pixels * tangent
            mid_tangent = sample(direction, midpoint)
            if mid_tangent is None or np.linalg.norm(mid_tangent) < 1e-4:
                break
            mid_tangent /= np.linalg.norm(mid_tangent) + 1e-8
            if np.dot(mid_tangent, tangent) < 0.0:
                mid_tangent *= -1.0
            next_point = point + step_pixels * mid_tangent
            if not valid(next_point):
                break
            curve.append(next_point.copy())
            point = next_point
            tangent = mid_tangent
        return curve

    grid_y, grid_x = np.mgrid[0:height:seed_spacing, 0:width:seed_spacing]
    grid_candidates = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    grid_candidates = grid_candidates[
        hair_mask[grid_candidates[:, 0], grid_candidates[:, 1]]
    ]
    all_candidates = np.argwhere(hair_mask)
    candidates = np.concatenate([grid_candidates, all_candidates], axis=0)

    visited = np.zeros_like(hair_mask, dtype=bool)
    mark_radius = max(1, seed_spacing // 2)
    curves_2d = []
    for y, x in candidates:
        if visited[y, x]:
            continue
        seed = np.array([x, y], dtype=np.float32)
        backward = follow(seed, -1.0)
        forward = follow(seed, 1.0)
        curve = np.asarray(backward[:0:-1] + forward, dtype=np.float32)
        if len(curve) < min_curve_points:
            visited[y, x] = True
            continue
        curves_2d.append(curve)
        pixels = np.rint(curve).astype(np.int32)
        for px, py in pixels:
            x0, x1 = max(0, px - mark_radius), min(width, px + mark_radius + 1)
            y0, y1 = max(0, py - mark_radius), min(height, py + mark_radius + 1)
            visited[y0:y1, x0:x1] = True

    inverse_calib = np.linalg.inv(np.asarray(calib, dtype=np.float64))
    curves_3d = []
    for curve in curves_2d:
        depths = np.asarray([sample(depth_map, point) for point in curve])
        clip_points = np.column_stack(
            [
                curve[:, 0] / max(width - 1, 1) * 2.0 - 1.0,
                curve[:, 1] / max(height - 1, 1) * 2.0 - 1.0,
                depths,
                np.ones(len(curve), dtype=np.float64),
            ]
        )
        world_h = clip_points @ inverse_calib.T
        world = world_h[:, :3] / np.clip(world_h[:, 3:4], 1e-8, None)
        curves_3d.append(world.astype(np.float32))

    return curves_3d


def backproject_direction(dx_2d, dy_2d, R_v, dz_dx, dz_dy):
    """Back-project a 2D strand direction into 3D world space.

    Args:
        dx_2d, dy_2d: (N,) float32, 2D direction in image plane (dx is right, dy is UP)
        R_v: (3, 3) float32, camera rotation matrix (world→cam)
        dz_dx, dz_dy: (N,) float32, depth gradients at each pixel

    Returns:
        dir_3d: (N, 3) float32, normalized 3D direction in world space
    """
    N = len(dx_2d)

    # In camera space, X and Y match dx_2d and dy_2d (since dy is already mapped to UP).
    # The surface depth Z is a function of (x, y).
    # The tangent vector in 3D along (dx, dy) has a Z component: dz = ∂Z/∂x * dx + ∂Z/∂y * dy
    # Note: dz_dy from the image is based on image Y (down), but dy_2d is UP.
    # We must be careful: if dy_2d is UP, it corresponds to a negative step in image Y.
    # So dz from dy_2d is dz_dy * (-dy_2d).
    # Wait, let's just use the exact math from LaplacePDEStrategy:
    # strand_dz = strand_dx * dz_dx + strand_dy * dz_dy
    # In LaplacePDEStrategy, strand_dy is UP (negated from image), and dz_dy is gradient along image Y (down).
    # So the dot product actually works out exactly the same as LaplacePDEStrategy.
    depth_grad = (dx_2d * dz_dx + dy_2d * dz_dy)

    # Camera-space direction
    d_cam = np.stack([dx_2d, dy_2d, depth_grad], axis=1)  # (N, 3)

    # Normalize
    norm = np.linalg.norm(d_cam, axis=1, keepdims=True) + 1e-8
    d_cam = d_cam / norm

    # World-space: d_world = R^T @ d_cam
    d_world = (R_v.T @ d_cam.T).T  # (N, 3)

    # Re-normalize
    norm = np.linalg.norm(d_world, axis=1, keepdims=True) + 1e-8
    d_world = d_world / norm

    return d_world


def backproject_direction_on_mesh(dx_2d, dy_2d, normals_world, R_v):
    """Lift 2D strand directions onto the tangent plane of a known mesh.

    For an orthographic camera, ``dx*ex + dy*ey + lambda*ez`` must be
    perpendicular to the camera-space surface normal. Near silhouettes the
    equation is ill-conditioned, so those samples are rejected.
    """
    normals_cam = (R_v @ normals_world.T).T
    denom = normals_cam[:, 2]
    stable = np.abs(denom) >= 0.15
    depth_component = np.zeros_like(dx_2d, dtype=np.float32)
    depth_component[stable] = -(
        normals_cam[stable, 0] * dx_2d[stable]
        + normals_cam[stable, 1] * dy_2d[stable]
    ) / denom[stable]

    direction_cam = np.stack([dx_2d, dy_2d, depth_component], axis=1)
    direction_world = (R_v.T @ direction_cam.T).T
    norms = np.linalg.norm(direction_world, axis=1, keepdims=True) + 1e-8
    return (direction_world / norms).astype(np.float32), stable


def compute_root_head_visibility(
    head_mesh_path, roots_world, calib, surface_tolerance=0.005
):
    """Return roots visible from an orthographic camera without head occlusion.

    Rays start on a camera-facing plane and travel toward each root.  A root is
    visible when the first head-mesh hit is at the root (within tolerance),
    behind it, or absent.  Back-side roots therefore cannot borrow a front-side
    hair-mask pixel merely because both project to the same image coordinate.
    """
    import open3d as o3d

    roots = np.asarray(roots_world, dtype=np.float32).reshape(-1, 3)
    if len(roots) == 0:
        return np.zeros(0, dtype=bool)

    mesh = o3d.io.read_triangle_mesh(str(head_mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Head mesh is empty: {head_mesh_path}")

    linear = np.asarray(calib, dtype=np.float32)[:3, :3]
    try:
        toward_camera = np.linalg.solve(
            linear, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
    except np.linalg.LinAlgError:
        toward_camera = linear[2]
    toward_camera /= np.linalg.norm(toward_camera) + 1e-8

    bounds = mesh.get_axis_aligned_bounding_box()
    ray_length = max(float(np.linalg.norm(bounds.get_extent())) * 3.0, 1.0)
    origins = roots + toward_camera[None, :] * ray_length
    directions = np.broadcast_to(-toward_camera, origins.shape).copy()
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hit_distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return (~np.isfinite(hit_distance)) | (
        hit_distance >= ray_length - float(surface_tolerance)
    )


def build_mesh_root_guidance(
    mesh_path, roots_world, calibs, strand_maps, head_mesh_path=None
):
    """Build mesh→root labels and per-view root visibility gates."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Hair mesh is empty: {mesh_path}")
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    roots = np.asarray(roots_world, dtype=np.float32).reshape(-1, 3)

    root_tree = cKDTree(roots)
    _, vertex_root_ids = root_tree.query(vertices, k=1, workers=-1)
    vertex_tree = cKDTree(vertices)

    root_homogeneous = np.column_stack([roots, np.ones(len(roots), dtype=np.float32)])
    visible_roots = {}
    for view, strand_map in strand_maps.items():
        if view not in calibs:
            continue
        calib = calibs[view][0]
        projected = root_homogeneous @ calib.T
        ndc = projected[:, :2] / np.clip(projected[:, 3:4], 1e-8, None)
        height, width = strand_map.shape[:2]
        px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        visible = np.zeros(len(roots), dtype=bool)
        indices = np.where(inside)[0]
        visible[indices] = strand_map[py[indices], px[indices], 0] > 0.1
        if head_mesh_path is not None:
            visible &= compute_root_head_visibility(
                head_mesh_path, roots, calib
            )
        visible_roots[view] = visible
        print(f"[MeshRootGuidance] {view}: {visible.sum()}/{len(roots)} roots covered")

    return {
        "roots": roots,
        "vertices": vertices,
        "normals": normals,
        "vertex_tree": vertex_tree,
        "vertex_root_ids": np.asarray(vertex_root_ids, dtype=np.int32),
        "visible_roots": visible_roots,
    }


def save_mesh_root_projections(
    guidance, calibs, strand_maps, background_paths, output_dir
):
    """Save per-view visible mesh regions colored by their assigned roots."""
    import os
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    vertices = guidance["vertices"]
    roots = guidance["roots"]
    vertex_root_ids = guidance["vertex_root_ids"]
    vertices_h = np.column_stack(
        [vertices, np.ones(len(vertices), dtype=np.float32)]
    )
    roots_h = np.column_stack([roots, np.ones(len(roots), dtype=np.float32)])

    root_min = roots.min(axis=0)
    root_span = np.maximum(roots.max(axis=0) - root_min, 1e-6)
    root_rgb = np.clip((roots - root_min) / root_span * 255.0, 0, 255).astype(np.uint8)
    root_bgr = root_rgb[:, ::-1]

    for view, strand_map in strand_maps.items():
        if view not in calibs or view not in guidance["visible_roots"]:
            continue
        height, width = strand_map.shape[:2]
        background = cv2.imread(str(background_paths.get(view, "")), cv2.IMREAD_COLOR)
        if background is None:
            background = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            background = cv2.resize(background, (width, height))

        calib = calibs[view][0]
        projected = vertices_h @ calib.T
        ndc = projected[:, :2] / np.clip(projected[:, 3:4], 1e-8, None)
        px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)

        # Orthographic normalized depth uses larger z for points nearer camera.
        inside_ids = np.where(inside)[0]
        flat = py[inside_ids] * width + px[inside_ids]
        depth = projected[inside_ids, 2]
        zbuffer = np.full(height * width, -np.inf, dtype=np.float32)
        np.maximum.at(zbuffer, flat, depth)
        visible = depth >= zbuffer[flat] - 0.003
        visible_ids = inside_ids[visible]

        layer = np.zeros_like(background)
        mesh_mask = np.zeros((height, width), dtype=np.uint8)
        colors = root_bgr[vertex_root_ids[visible_ids]]
        layer[py[visible_ids], px[visible_ids]] = colors
        mesh_mask[py[visible_ids], px[visible_ids]] = 255
        mesh_mask = cv2.dilate(mesh_mask, np.ones((3, 3), np.uint8), iterations=1)
        layer = cv2.dilate(layer, np.ones((3, 3), np.uint8), iterations=1)

        overlay = background.copy()
        colored = mesh_mask > 0
        blended = cv2.addWeighted(background, 0.35, layer, 0.65, 0)
        overlay[colored] = blended[colored]

        root_projected = roots_h @ calib.T
        root_ndc = root_projected[:, :2] / np.clip(
            root_projected[:, 3:4], 1e-8, None
        )
        root_px = np.rint((root_ndc[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
        root_py = np.rint((root_ndc[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
        root_visible = guidance["visible_roots"][view]
        root_inside = (
            root_visible
            & (root_px >= 0) & (root_px < width)
            & (root_py >= 0) & (root_py < height)
        )
        overlay[root_py[root_inside], root_px[root_inside]] = (255, 255, 255)
        cv2.putText(
            overlay,
            f"{view}: visible roots {root_inside.sum()}/{len(roots)}",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imwrite(os.path.join(output_dir, f"{view}.png"), overlay)


def compute_depth_gradient(depth_map, px, py):
    from scipy.ndimage import gaussian_filter
    H, W = depth_map.shape

    # Smooth depth to get meaningful gradients at strand scale
    depth_smooth = gaussian_filter(depth_map, sigma=3.0)

    # Central-difference gradient in normalized depth per pixel
    valid = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
    dz_dx = np.zeros(len(px), dtype=np.float32)
    dz_dy = np.zeros(len(px), dtype=np.float32)

    idx = np.where(valid)[0]
    dz_dx[idx] = (depth_smooth[py[idx], px[idx] + 1] - depth_smooth[py[idx], px[idx] - 1]) * 0.5
    dz_dy[idx] = (depth_smooth[py[idx] + 1, px[idx]] - depth_smooth[py[idx] - 1, px[idx]]) * 0.5

    return dz_dx * 256.0, dz_dy * 256.0


def build_visible_mesh_surface_shell(
    mesh_path,
    calibs,
    seg_masks,
    b_min,
    b_max,
    resolution,
    shell_iterations=2,
):
    """Voxelize first-visible mesh hits from each hair-segmented view.

    This deliberately does not classify points with the bald-head SDF.  The
    source images constrain only the visible outer surface, so each pixel
    contributes the first ray hit on ``mesh_path`` and the per-view hair seg
    decides whether that hit belongs to the observed hair surface.
    """
    import open3d as o3d
    from scipy.ndimage import binary_dilation

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Surface-shell mesh is empty: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    b_min = np.asarray(b_min, dtype=np.float64)
    b_max = np.asarray(b_max, dtype=np.float64)
    shape = np.full(3, int(resolution), dtype=np.int32)
    shell = np.zeros(tuple(shape), dtype=bool)
    per_view_counts = {}

    for view, mask in seg_masks.items():
        if view not in calibs:
            continue
        calib = np.asarray(calibs[view][0], dtype=np.float64)
        height, width = mask.shape
        py, px = np.nonzero(mask)
        if not len(px):
            per_view_counts[view] = 0
            continue
        uv = np.column_stack([
            px / max(width - 1, 1) * 2.0 - 1.0,
            py / max(height - 1, 1) * 2.0 - 1.0,
        ])
        inverse = np.linalg.inv(calib)

        def unproject(depth):
            clip = np.column_stack([
                uv,
                np.full(len(uv), depth, dtype=np.float64),
                np.ones(len(uv), dtype=np.float64),
            ])
            world_h = clip @ inverse.T
            return world_h[:, :3] / world_h[:, 3:4]

        near = unproject(1.5)
        far = unproject(-0.5)
        direction = far - near
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        rays = np.column_stack([near, direction]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        valid = np.isfinite(hit)
        points = near[valid] + direction[valid] * hit[valid, None]
        grid = np.rint(
            (points - b_min) / np.maximum(b_max - b_min, 1e-12)
            * (shape - 1)
        ).astype(np.int32)
        inside = np.all((grid >= 0) & (grid < shape), axis=1)
        grid = grid[inside]
        shell[grid[:, 0], grid[:, 1], grid[:, 2]] = True
        per_view_counts[view] = int(len(grid))

    raw_shell = shell.copy()
    if shell_iterations > 0:
        shell = binary_dilation(
            shell,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(shell_iterations),
        )
    return shell, raw_shell, per_view_counts


def extract_view_mesh_surface_contribution(
    mesh_path,
    strand_map,
    seg_mask,
    calib,
    pixel_stride=2,
    vector_length=0.003,
):
    """Lift one view's strand directions onto its first-visible mesh surface.

    The returned short line segments are visualization/debug boundary vectors,
    not independently back-projected hair curves.  Position comes from the
    first camera-ray hit, while direction comes solely from this view's strand
    map and is projected onto the hit triangle's tangent plane.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Contribution mesh is empty: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    strand_map = np.asarray(strand_map, dtype=np.float32)
    seg_mask = np.asarray(seg_mask, dtype=bool)
    height, width = seg_mask.shape
    valid_pixels = seg_mask & (strand_map[:, :, 0] > 0.1)
    py, px = np.nonzero(valid_pixels)
    keep = (px % pixel_stride == 0) & (py % pixel_stride == 0)
    px, py = px[keep], py[keep]
    if not len(px):
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.int32)

    matrix = np.asarray(calib[0] if isinstance(calib, tuple) else calib, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    uv = np.column_stack([
        px / max(width - 1, 1) * 2.0 - 1.0,
        py / max(height - 1, 1) * 2.0 - 1.0,
    ])

    def unproject(coords, depth):
        clip = np.column_stack([coords, depth, np.ones(len(coords))])
        world_h = clip @ inverse.T
        return world_h[:, :3] / world_h[:, 3:4]

    near = unproject(uv, np.full(len(uv), 1.5))
    far = unproject(uv, np.full(len(uv), -0.5))
    ray_direction = far - near
    ray_direction /= np.linalg.norm(ray_direction, axis=1, keepdims=True) + 1e-12
    result = scene.cast_rays(
        o3d.core.Tensor(np.column_stack([near, ray_direction]).astype(np.float32))
    )
    hit = result["t_hit"].numpy()
    ray_valid = np.isfinite(hit)
    points = near[ray_valid] + ray_direction[ray_valid] * hit[ray_valid, None]
    normals = result["primitive_normals"].numpy()[ray_valid].astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    uv_hit = uv[ray_valid]
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ matrix.T
    depth = projected[:, 2] / projected[:, 3]
    epsilon = 2.0 / max(height, width)
    tangent_u = unproject(uv_hit + np.array([epsilon, 0.0]), depth) - points
    tangent_v = unproject(uv_hit + np.array([0.0, epsilon]), depth) - points
    hit_px, hit_py = px[ray_valid], py[ray_valid]
    dx = 1.0 - 2.0 * strand_map[hit_py, hit_px, 2]
    dy = 2.0 * strand_map[hit_py, hit_px, 1] - 1.0
    directions = dx[:, None] * tangent_u + dy[:, None] * tangent_v
    directions -= np.sum(directions * normals, axis=1, keepdims=True) * normals
    norm = np.linalg.norm(directions, axis=1)
    stable = norm > 1e-8
    points = points[stable].astype(np.float32)
    directions = (directions[stable] / norm[stable, None]).astype(np.float32)
    endpoints = points + float(vector_length) * directions
    count = len(points)
    line_points = np.vstack([points, endpoints])
    lines = np.column_stack([np.arange(count), np.arange(count) + count]).astype(np.int32)
    return line_points, lines


def build_multiview_seg_support_volume(
    calibs,
    seg_masks,
    b_min,
    b_max,
    resolution,
    slab_size=8,
    head_mesh_path="data/head_model.obj",
    occluder_mesh_path=None,
    visibility_tolerance=None,
):
    """Fuse per-view segmentations with head-aware visibility.

    A view contributes positive evidence when a voxel is visible in front of
    the head and projects inside that view's hair segmentation. Visible
    background is negative evidence, while voxels behind the head or the first
    visible hair surface are ignored for that view. Front defines the
    authoritative visible hairstyle envelope; where front is occluded or
    unavailable, side/back votes provide support.
    """
    import open3d as o3d

    resolution = int(resolution)
    b_min = np.asarray(b_min, dtype=np.float32)
    b_max = np.asarray(b_max, dtype=np.float32)
    if visibility_tolerance is None:
        voxel_size = np.linalg.norm(
            (b_max - b_min) / max(resolution - 1, 1)
        )
        visibility_tolerance = max(1.5 * voxel_size, 0.03)

    head_mesh = o3d.io.read_triangle_mesh(str(head_mesh_path))
    if not head_mesh.has_vertices() or not head_mesh.has_triangles():
        raise ValueError(f"Head mesh is empty: {head_mesh_path}")
    occlusion_scene = o3d.t.geometry.RaycastingScene()
    occlusion_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(head_mesh)
    )
    if occluder_mesh_path is not None:
        occluder_mesh = o3d.io.read_triangle_mesh(str(occluder_mesh_path))
        if not occluder_mesh.has_vertices() or not occluder_mesh.has_triangles():
            raise ValueError(f"Occluder mesh is empty: {occluder_mesh_path}")
        occlusion_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(occluder_mesh)
        )

    view_visibility = {}
    for view, mask in seg_masks.items():
        if view not in calibs:
            continue
        calib = calibs[view][0] if isinstance(calibs[view], tuple) else calibs[view]
        inverse = np.linalg.inv(np.asarray(calib, dtype=np.float64))
        height, width = mask.shape
        py, px = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        uv = np.column_stack([
            px.ravel() / max(width - 1, 1) * 2.0 - 1.0,
            py.ravel() / max(height - 1, 1) * 2.0 - 1.0,
        ])

        def unproject(depth):
            clip = np.column_stack([
                uv,
                np.full(len(uv), depth, dtype=np.float64),
                np.ones(len(uv), dtype=np.float64),
            ])
            world_h = clip @ inverse.T
            return world_h[:, :3] / world_h[:, 3:4]

        ray_origins = unproject(1.5)
        ray_directions = unproject(-0.5) - ray_origins
        ray_directions /= (
            np.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-12
        )
        rays = np.column_stack([ray_origins, ray_directions]).astype(np.float32)
        occluder_hits = occlusion_scene.cast_rays(
            o3d.core.Tensor(rays)
        )["t_hit"].numpy()
        view_visibility[view] = (
            ray_origins.astype(np.float32).reshape(height, width, 3),
            ray_directions.astype(np.float32).reshape(height, width, 3),
            occluder_hits.reshape(height, width),
        )

    xs = np.linspace(b_min[0], b_max[0], resolution, dtype=np.float32)
    ys = np.linspace(b_min[1], b_max[1], resolution, dtype=np.float32)
    zs = np.linspace(b_min[2], b_max[2], resolution, dtype=np.float32)
    support = np.zeros((resolution, resolution, resolution), dtype=bool)
    positive_count = 0
    negative_count = 0

    for x_start in range(0, resolution, slab_size):
        x_stop = min(x_start + slab_size, resolution)
        gx, gy, gz = np.meshgrid(xs[x_start:x_stop], ys, zs, indexing="ij")
        points = np.column_stack([
            gx.ravel(), gy.ravel(), gz.ravel(), np.ones(gx.size, dtype=np.float32)
        ])
        front_positive = np.zeros(len(points), dtype=bool)
        front_negative = np.zeros(len(points), dtype=bool)
        side_positive_votes = np.zeros(len(points), dtype=np.uint8)
        side_negative_votes = np.zeros(len(points), dtype=np.uint8)
        for view, mask in seg_masks.items():
            if view not in calibs or view not in view_visibility:
                continue
            calib = calibs[view][0] if isinstance(calibs[view], tuple) else calibs[view]
            projected = points @ np.asarray(calib, dtype=np.float32).T
            denominator = projected[:, 3:4]
            denominator = np.where(
                np.abs(denominator) < 1e-8,
                np.copysign(1e-8, denominator + 1e-12),
                denominator,
            )
            uv = projected[:, :2] / denominator
            height, width = mask.shape
            px = np.rint((uv[:, 0] + 1.0) * 0.5 * (width - 1)).astype(np.int32)
            py = np.rint((uv[:, 1] + 1.0) * 0.5 * (height - 1)).astype(np.int32)
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            ids = np.flatnonzero(inside)
            if not len(ids):
                continue

            origins, directions, head_hits = view_visibility[view]
            pixel_y, pixel_x = py[ids], px[ids]
            ray_origins = origins[pixel_y, pixel_x]
            ray_directions = directions[pixel_y, pixel_x]
            voxel_distance = np.sum(
                (points[ids, :3] - ray_origins) * ray_directions,
                axis=1,
            )
            head_distance = head_hits[pixel_y, pixel_x]
            visible = (
                (voxel_distance >= 0.0)
                & (
                    ~np.isfinite(head_distance)
                    | (
                        voxel_distance
                        <= head_distance + float(visibility_tolerance)
                    )
                )
            )
            is_hair = np.asarray(mask[pixel_y, pixel_x], dtype=bool)
            positive = visible & is_hair
            negative = visible & ~is_hair
            if view == "front":
                front_positive[ids] |= positive
                front_negative[ids] |= negative
            else:
                side_positive_votes[ids] += positive.astype(np.uint8)
                side_negative_votes[ids] += negative.astype(np.uint8)

        # Front is the authoritative hairstyle envelope. Where the head hides
        # a voxel from front, any visible side/back positive may recover it.
        # Side-view negatives are not authoritative because synthesized masks
        # and the aligned head/hair geometry differ by a few centimeters.
        side_consensus = side_positive_votes > 0
        slab_support = front_positive | (~front_negative & side_consensus)
        positive_count += int(
            (front_positive | (side_positive_votes > 0)).sum()
        )
        negative_count += int(
            (front_negative | (side_negative_votes > 0)).sum()
        )
        support[x_start:x_stop] = slab_support.reshape(
            x_stop - x_start, resolution, resolution
        )
    print(
        "[MultiviewFusion] Visibility-aware seg support: "
        f"positive={positive_count}, negative={negative_count}, "
        f"accepted={int(support.sum())}"
    )
    return support


def build_mesh_metric_band(
    mesh_path,
    support_volume,
    b_min,
    b_max,
    band_width=0.05,
    slab_size=8,
):
    """Build a metric-width mesh band inside a precomputed support volume."""
    import open3d as o3d

    support_volume = np.asarray(support_volume, dtype=bool)
    if support_volume.ndim != 3:
        raise ValueError(
            f"support_volume must be 3D, got {support_volume.shape}"
        )
    if band_width <= 0.0:
        raise ValueError(f"band_width must be positive, got {band_width}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Metric-band mesh is empty: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    b_min = np.asarray(b_min, dtype=np.float32)
    b_max = np.asarray(b_max, dtype=np.float32)
    shape = np.asarray(support_volume.shape, dtype=np.int32)
    axes = [
        np.linspace(b_min[axis], b_max[axis], shape[axis], dtype=np.float32)
        for axis in range(3)
    ]
    band = np.zeros(tuple(shape), dtype=bool)
    candidate_count = 0

    for x_start in range(0, shape[0], int(slab_size)):
        x_stop = min(x_start + int(slab_size), shape[0])
        local_indices = np.argwhere(support_volume[x_start:x_stop])
        if not len(local_indices):
            continue
        global_indices = local_indices.copy()
        global_indices[:, 0] += x_start
        points = np.column_stack([
            axes[0][global_indices[:, 0]],
            axes[1][global_indices[:, 1]],
            axes[2][global_indices[:, 2]],
        ]).astype(np.float32)
        distances = scene.compute_distance(
            o3d.core.Tensor(points)
        ).numpy()
        accepted = global_indices[distances < float(band_width)]
        band[accepted[:, 0], accepted[:, 1], accepted[:, 2]] = True
        candidate_count += len(global_indices)

    print(
        "[MultiviewFusion] Metric hair band: "
        f"width={band_width:.3f}m, candidates={candidate_count}, "
        f"accepted={int(band.sum())}"
    )
    return band


def fuse_multiview_orientation(
    strand_maps,      # dict: view → (H, W, 3) float32 [0, 1]
    depth_maps,        # dict: view → (H, W) float32 [0, 1]
    calibs,            # dict: view → (4, 4) float32
    resolution=256,
    b_min=None,
    b_max=None,
    center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    extent=0.3,
    front_surface_margin=0.02,
    other_surface_margin=0.015,
    mesh_root_guidance=None,
):
    """Fuse multi-view 2D strand directions into a 3D orientation volume.

    Hard rule: front view claims EXCLUSIVE ownership of any voxel it can see.
    Other views only contribute where front cannot see.

    Args:
        strand_maps: {view: (512, 512, 3) float32}
        depth_maps:  {view: (512, 512) float32}
        calibs:      {view: (4, 4) float32}
        resolution:  voxel grid resolution
        b_min, b_max: world-space bounding box (auto-computed if None)
        center:      mesh center (world coords)
        extent:      mesh extent (bounding box diagonal)
        front_surface_margin: depth tolerance for front surface voxels
        other_surface_margin: depth tolerance for other views

    Returns:
        orien_vol:      (3, R, R, R) float32, fused 3D orientation field
        boundary_mask:  (R, R, R) bool, which voxels have explicit direction
        view_ownership: (R, R, R) int8, which view owns each voxel
                        (0=front, 1=left, 2=right, 3=back, -1=unseen)
    """
    R = resolution
    view_names = ["front", "left", "right", "back"]

    # Auto-compute bounding box
    if b_min is None:
        b_min = center - extent * 1.2
    if b_max is None:
        b_max = center + extent * 1.2
    b_min = np.asarray(b_min, dtype=np.float32)
    b_max = np.asarray(b_max, dtype=np.float32)

    H = W = 512  # strand_map / depth_map resolution

    # Voxel grid — use meshgrid for full (R,R,R) coordinates
    xs = np.linspace(b_min[0], b_max[0], R, dtype=np.float32)
    ys = np.linspace(b_min[1], b_max[1], R, dtype=np.float32)
    zs = np.linspace(b_min[2], b_max[2], R, dtype=np.float32)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')  # each (R, R, R)
    vox_flat = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=0)  # (3, R^3)
    del gx, gy, gz

    # Output volumes
    orien_vol = np.zeros((3, R, R, R), dtype=np.float32)
    weight_vol = np.zeros((R, R, R), dtype=np.float32)
    view_ownership = np.full((R, R, R), -1, dtype=np.int8)
    view_index = {"front": 0, "left": 1, "right": 2, "back": 3}

    # ── Pass 1: Front view FIRST (if available) ─────────────────────
    if "front" in strand_maps and "front" in calibs:
        print("[MultiviewFusion] Pass 1/2: Front view (exclusive ground truth)...")
        v = "front"
        calib, R_pure = calibs[v]  # calib=4×4 projection, R_pure=3×3 rotation
        P3 = calib[:3, :3]   # includes intrinsic scaling — for projection
        t3 = calib[:3, 3:4]

        # Project all voxels to front camera (uses full projection for NDC)
        pts_cam = (P3 @ vox_flat) + t3  # (3, R^3)
        vox_depth = pts_cam[2]  # depth in camera space

        # UV in [-1, 1] NDC
        u = pts_cam[0]
        v_uv = pts_cam[1]
        px_front = np.clip(((u + 1.0) * 0.5 * W).astype(np.int32), 0, W - 1)
        py_front = np.clip(((v_uv + 1.0) * 0.5 * H).astype(np.int32), 0, H - 1)

        # Query front strand_map and depth_map
        dx_f, dy_f, hair_f = decode_strand_2d(strand_maps["front"], px_front, py_front)
        surf_depth_f = np.where(
            (px_front >= 0) & (px_front < W) & (py_front >= 0) & (py_front < H),
            depth_maps["front"][py_front, px_front],
            -1.0,
        )

        # Surface margin for depth comparison
        margin = (b_max[2] - b_min[2]) / R * 2.0 + front_surface_margin
        is_surface_f = hair_f & (np.abs(vox_depth - surf_depth_f) < margin)

        # Back-project front 2D direction → 3D
        dz_dx_f, dz_dy_f = compute_depth_gradient(depth_maps["front"], px_front, py_front)
        dir_3d_f = backproject_direction(dx_f, dy_f, R_pure, dz_dx_f, dz_dy_f)

        # Assign front direction where front can see
        idx_surf = np.where(is_surface_f)[0]
        for c in range(3):
            orien_vol.ravel()[c * R**3 + idx_surf] = dir_3d_f[idx_surf, c]
        weight_vol.ravel()[idx_surf] = 1.0
        view_ownership.ravel()[idx_surf] = view_index["front"]

        n_front = len(idx_surf)
        print(f"  Front covers {n_front}/{R**3} voxels ({100*n_front/R**3:.1f}%)")
    else:
        print("[MultiviewFusion] Pass 1/2: Front view NOT provided. Proceeding with available views...")

    # ── Pass 2: Other views (fill gaps only) ─────────────────────
    print("[MultiviewFusion] Pass 2/2: Left/Right/Back (fill unseen)...")
    view_weights = {"left": 1.0, "right": 1.0, "back": 0.5}

    valid_other_views = [v for v in ["left", "right", "back"] if v in calibs and v in strand_maps]
    for v in valid_other_views:
        calib, R_pure = calibs[v]  # (calib_4x4, R_3x3)
        P3 = calib[:3, :3]
        t3 = calib[:3, 3:4]

        pts_cam = (P3 @ vox_flat) + t3
        vox_depth = pts_cam[2]
        u = pts_cam[0]
        v_uv = pts_cam[1]
        px = np.clip(((u + 1.0) * 0.5 * W).astype(np.int32), 0, W - 1)
        py = np.clip(((v_uv + 1.0) * 0.5 * H).astype(np.int32), 0, H - 1)

        dx_v, dy_v, hair_v = decode_strand_2d(strand_maps[v], px, py)
        surf_depth_v = np.where(
            (px >= 0) & (px < W) & (py >= 0) & (py < H),
            depth_maps[v][py, px],
            -1.0,
        )

        margin = (b_max[2] - b_min[2]) / R * 2.0 + other_surface_margin
        is_surface_v = hair_v & (np.abs(vox_depth - surf_depth_v) < margin)

        # KEY: only fill voxels NOT already owned by front
        already_owned = weight_vol.ravel() > 0.0
        fill_mask = is_surface_v & ~already_owned

        if fill_mask.sum() == 0:
            print(f"  {v}: 0 new voxels (all already covered)")
            continue

        w = view_weights[v]
        idx_candidates = np.where(fill_mask)[0]

        if mesh_root_guidance is not None and v in mesh_root_guidance["visible_roots"]:
            voxel_size = np.linalg.norm((b_max - b_min) / max(R - 1, 1))
            query_points = vox_flat[:, idx_candidates].T
            distances, vertex_ids = mesh_root_guidance["vertex_tree"].query(
                query_points, k=1, workers=-1
            )
            root_ids = mesh_root_guidance["vertex_root_ids"][vertex_ids]
            root_visible = mesh_root_guidance["visible_roots"][v][root_ids]
            near_mesh = distances <= max(2.5 * voxel_size, 0.004)
            normals = mesh_root_guidance["normals"][vertex_ids]
            candidate_dirs, tangent_stable = backproject_direction_on_mesh(
                dx_v[idx_candidates], dy_v[idx_candidates], normals, R_pure
            )
            accepted = near_mesh & root_visible & tangent_stable
            idx_fill = idx_candidates[accepted]
            dir_3d_v = np.zeros((len(px), 3), dtype=np.float32)
            dir_3d_v[idx_fill] = candidate_dirs[accepted]
            print(
                f"  {v} mesh/root gate: {len(idx_fill)}/{len(idx_candidates)} accepted "
                f"(mesh={near_mesh.sum()}, roots={root_visible.sum()}, tangent={tangent_stable.sum()})"
            )
        else:
            dz_dx_v, dz_dy_v = compute_depth_gradient(depth_maps[v], px, py)
            dir_3d_v = backproject_direction(dx_v, dy_v, R_pure, dz_dx_v, dz_dy_v)
            idx_fill = idx_candidates

        if len(idx_fill) == 0:
            print(f"  {v}: 0 voxels after mesh/root gating")
            continue

        for c in range(3):
            orien_vol.ravel()[c * R**3 + idx_fill] += w * dir_3d_v[idx_fill, c]
        weight_vol.ravel()[idx_fill] += w
        view_ownership.ravel()[idx_fill] = view_index[v]

        print(f"  {v}: {len(idx_fill)} new voxels (weight={w})")

    # ── Normalize ────────────────────────────────────────────────
    boundary_mask = weight_vol > 0.0
    for c in range(3):
        chan = orien_vol[c]
        chan[boundary_mask] /= weight_vol[boundary_mask]
        orien_vol[c] = chan

    # Normalize direction vectors
    norm = np.sqrt(np.sum(orien_vol ** 2, axis=0)) + 1e-8
    orien_vol = orien_vol / norm

    n_total = boundary_mask.sum()
    print(f"[MultiviewFusion] Total surface voxels: {n_total}/{R**3} "
          f"({100*n_total/R**3:.1f}%)")

    # Per-view breakdown
    for v in view_names:
        vi = view_index[v]
        n_v = (view_ownership == vi).sum()
        print(f"  {v}: {n_v} voxels")

    return orien_vol, boundary_mask, view_ownership
