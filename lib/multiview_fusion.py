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

    # ── Pass 1: Front view FIRST (exclusive) ─────────────────────
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

        dz_dx_v, dz_dy_v = compute_depth_gradient(depth_maps[v], px, py)
        dir_3d_v = backproject_direction(dx_v, dy_v, R_pure, dz_dx_v, dz_dy_v)

        w = view_weights[v]
        idx_fill = np.where(fill_mask)[0]

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
