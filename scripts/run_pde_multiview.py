"""
Multi-View 3D Hair Synthesis via PDE.

Loads 4-view strand_maps + depth_maps, fuses them into a 3D orientation
volume (front=exclusive ground truth), solves Laplace PDE, and traces
strands via RK4 integration.

Usage:
  python scripts/run_pde_multiview.py \
      --strand_dir  results/multiview_strand_depth/strand_map \
      --depth_dir   results/multiview_strand_depth/depth_map \
      --mesh_obj    results/test_pixal3d/hair_mesh_flame_extracted.obj \
      --out_ply     results/multiview_pde/hair.ply
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import numpy as np
import torch
import cv2
import trimesh
import open3d as o3d
import imageio.v2 as imageio

from lib.multiview_fusion import (
    build_blender_calib,
    fuse_multiview_orientation,
)
from lib.multiview_pde import MultiViewLaplacePDEStrategy


def main():
    parser = argparse.ArgumentParser(
        description="Multi-View 3D Hair PDE Synthesis"
    )
    parser.add_argument("--strand_dir",
                        default="results/multiview_strand_depth/strand_map")
    parser.add_argument("--depth_dir",
                        default="results/multiview_strand_depth/depth_map")
    parser.add_argument("--mesh_obj",
                        default="results/test_pixal3d/hair_mesh_flame_extracted.obj")
    parser.add_argument("--out_ply",
                        default="results/multiview_pde/hair.ply")
    parser.add_argument("--roots",
                        default="data/roots10k.obj")
    parser.add_argument("--pde_resolution", type=int, default=256)
    parser.add_argument("--pde_dilation_iters", type=int, default=6)
    parser.add_argument("--pde_cg_tol", type=float, default=1e-4)
    parser.add_argument("--pde_cg_maxiter", type=int, default=2000)
    parser.add_argument("--pde_anisotropy", type=float, default=0.8)
    parser.add_argument("--pde_alpha_mix", type=float, default=3.0,
                        help="2nd-order Laplacian weight (higher = better CG convergence)")
    parser.add_argument("--pde_beta_mix", type=float, default=1.0,
                        help="4th-order bilaplacian weight")
    parser.add_argument("--front_surface_margin", type=float, default=0.02,
                        help="Depth tolerance for front surface voxels")
    parser.add_argument("--other_surface_margin", type=float, default=0.015,
                        help="Depth tolerance for other views")
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--hair_unit", type=float, default=0.006)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    cuda = torch.device("cuda:0")

    # ================================================================
    #  1. Load multi-view strand_maps + depth_maps
    # ================================================================
    print("=" * 60)
    print("Step 1: Loading multi-view data...")
    views = ["front", "left", "right", "back"]
    strand_maps = {}
    depth_maps = {}

    for v in views:
        sp = os.path.join(args.strand_dir, f"{v}.png")
        dp = os.path.join(args.depth_dir, f"{v}.npy")

        strand_maps[v] = imageio.imread(sp).astype(np.float32) / 255.0
        depth_maps[v] = np.load(dp).astype(np.float32)
        print(f"  {v}: strand={strand_maps[v].shape}, depth={depth_maps[v].shape}")

    # ================================================================
    #  2. Build camera calibration matrices
    # ================================================================
    print("\nStep 2: Building camera calibrations...")

    # Load mesh to get extent & the REAL calib center
    mesh = o3d.io.read_triangle_mesh(args.mesh_obj)
    verts = np.asarray(mesh.vertices)
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    print(f"  Mesh: {len(verts)} verts, extent={extent:.4f}")

    # Load REAL front calibration — MUST match the strand_map coordinate system
    from scripts.recon3D import load_calib
    front_calib_path = "results/real_imgs/param/0a1ba3dbefc8934ab60577c5c91f66a0.npy"
    print(f"  Loading REAL front calib from: {front_calib_path}")
    # Load param for rotation + center BEFORE building calibs
    param = np.load(front_calib_path, allow_pickle=True).item()
    real_center = param.get('center').flatten().astype(np.float32)
    R_real = param.get('R').astype(np.float32)
    # Normalize rows — raw R includes model scale, not orthonormal
    R_real = R_real / (np.linalg.norm(R_real, axis=1, keepdims=True) + 1e-8)

    front_calib = load_calib(front_calib_path, loadSize=1024)  # original calib was for 1024
    if isinstance(front_calib, torch.Tensor):
        front_calib = front_calib.numpy()
    calibs = {"front": (front_calib.astype(np.float32), R_real)}
    print(f"  Real center: {real_center}")

    # Side/back: Blender cameras aligned to the same center
    for v in ["left", "right", "back"]:
        calibs[v] = build_blender_calib(v, real_center, extent)
    print(f"  Side cameras built with real center")

    # ================================================================
    #  3. Fuse multi-view strand directions → 3D orientation volume
    # ================================================================
    print(f"\nStep 3: Fusing multi-view data into 3D volume "
          f"(resolution={args.pde_resolution})...")

    # Bounding box: match original PDE — head_model + hair_mesh + padding
    head_mesh = o3d.io.read_triangle_mesh("data/head_model.obj")
    head_bbox = head_mesh.get_axis_aligned_bounding_box()
    b_min = head_bbox.get_min_bound()
    b_max = head_bbox.get_max_bound()
    hair_bbox = mesh.get_axis_aligned_bounding_box()
    b_min = np.minimum(b_min, hair_bbox.get_min_bound())
    b_max = np.maximum(b_max, hair_bbox.get_max_bound())
    padding = 0.03
    b_min = b_min - padding
    b_max = b_max + padding
    print(f"  BBox: [{b_min}, {b_max}]")

    fused_orien, boundary_mask, view_ownership = fuse_multiview_orientation(
        strand_maps=strand_maps,
        depth_maps=depth_maps,
        calibs=calibs,
        resolution=args.pde_resolution,
        b_min=b_min,
        b_max=b_max,
        center=real_center,
        extent=extent,
        front_surface_margin=args.front_surface_margin,
        other_surface_margin=args.other_surface_margin,
    )

    # Save fusion debug data (before dilation)
    np.savez_compressed(
        os.path.join(os.path.dirname(args.out_ply), "fusion_debug.npz"),
        fused_orien=fused_orien,
        boundary_mask=boundary_mask,
        view_ownership=view_ownership,
    )
    print(f"  Fusion debug saved.")

    # Build hair volume: dilate boundary + add scalp region from head model.
    # This ensures roots on the scalp have orientation to grow from.
    from scipy.ndimage import binary_dilation
    struct = np.ones((9, 9, 9), dtype=bool)
    hair_volume = binary_dilation(boundary_mask, structure=struct, iterations=4)
    hair_volume = hair_volume | boundary_mask

    # Add head model interior (scalp) to hair volume
    head_mesh_occ = o3d.io.read_triangle_mesh("data/head_model.obj")
    head_t = o3d.t.geometry.TriangleMesh.from_legacy(head_mesh_occ)
    head_scene = o3d.t.geometry.RaycastingScene()
    head_scene.add_triangles(head_t)
    R_vol = hair_volume.shape[0]
    xs_h = np.linspace(b_min[0], b_max[0], R_vol)
    ys_h = np.linspace(b_min[1], b_max[1], R_vol)
    zs_h = np.linspace(b_min[2], b_max[2], R_vol)
    gx, gy, gz = np.meshgrid(xs_h, ys_h, zs_h, indexing='ij')
    scalp_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    scalp_sdf = head_scene.compute_signed_distance(
        o3d.core.Tensor(scalp_pts, dtype=o3d.core.Dtype.Float32)
    ).numpy().reshape(R_vol, R_vol, R_vol)
    scalp_mask = scalp_sdf < 0.02  # inside or near head surface
    hair_volume = hair_volume | scalp_mask
    del gx, gy, gz, scalp_pts, scalp_sdf
    # head_scene kept alive for Step 4 scalp normal computation

    print(f"  Hair volume: {hair_volume.sum()} voxels "
          f"({100*hair_volume.sum()/hair_volume.size:.1f}%)")

    # ================================================================
    #  4. Build PDE strategy with fused boundary
    # ================================================================
    print("\nStep 4: Solving Laplace PDE on fused boundary...")

    # Use cv2 (BGR) to match original pipeline's channel order
    import cv2
    front_depth = depth_maps["front"]
    front_strand_bgr = cv2.imread(
        os.path.join(args.strand_dir, "front.png")
    ).astype(np.float32) / 255.0 * 2.0 - 1.0

    hairstep = np.concatenate([
        front_strand_bgr.transpose(2, 0, 1),  # (3, 512, 512) BGR in [-1,1]
        front_depth[None, :, :],
    ], axis=0)

    data = {
        "hairstep": torch.from_numpy(hairstep).float(),
        "calib": torch.from_numpy(calibs["front"][0]).float().clone(),
    }

    b_min_val = np.asarray(b_min, dtype=np.float32)
    b_max_val = np.asarray(b_max, dtype=np.float32)

    # Run ORIGINAL single-view LaplacePDEStrategy (converges reliably)
    from lib.recon_strategy.laplace_pde import LaplacePDEStrategy

    opt_pde = argparse.Namespace()
    opt_pde.pde_resolution = args.pde_resolution
    opt_pde.pde_dilation_iters = args.pde_dilation_iters
    opt_pde.pde_cg_tol = args.pde_cg_tol
    opt_pde.pde_cg_maxiter = args.pde_cg_maxiter
    opt_pde.pde_anisotropy = args.pde_anisotropy
    opt_pde.pde_alpha_mix = args.pde_alpha_mix
    opt_pde.pde_beta_mix = args.pde_beta_mix
    opt_pde.b_min = b_min_val
    opt_pde.b_max = b_max_val

    strategy = LaplacePDEStrategy(opt_pde, cuda)
    strategy.filter(data, mesh_path=args.mesh_obj)
    strategy.set_query_mode("orien")

    # PDE's occupancy only covers the front-visible surface.
    # Expand to hair_volume with EDT from nearest PDE direction.
    pde_nonzero = np.abs(strategy._orien_vol).sum(axis=0) > 1e-6
    missing = hair_volume & ~pde_nonzero
    if missing.sum() > 0:
        from scipy.ndimage import distance_transform_edt
        print(f"  Expanding to hair_volume: +{missing.sum()} voxels via EDT...")
        _, idx = distance_transform_edt(~pde_nonzero, return_indices=True)
        for c in range(3):
            flat = strategy._orien_vol[c].ravel()
            strategy._orien_vol[c][missing] = flat[idx[c].ravel()[missing.ravel()]]
        print(f"  Done.")

    # Save orientation volume for debugging
    np.save(
        os.path.join(os.path.dirname(args.out_ply), "debug_orien_vol.npy"),
        strategy._orien_vol,
    )
    print(f"  Orientation volume saved.")

    # ================================================================
    #  5. Build collision volume (head model)
    # ================================================================
    print("\nStep 5: Building collision SDF for head model...")
    head_mesh = o3d.io.read_triangle_mesh("data/head_model.obj")
    head_t = o3d.t.geometry.TriangleMesh.from_legacy(head_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(head_t)

    R = args.pde_resolution
    xs = np.linspace(b_min[0], b_max[0], R)
    ys = np.linspace(b_min[1], b_max[1], R)
    zs = np.linspace(b_min[2], b_max[2], R)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts_grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    del gx, gy, gz

    queries = o3d.core.Tensor(pts_grid, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(queries).numpy().reshape(R, R, R)

    dx_v = (b_max[0] - b_min[0]) / (R - 1)
    dy_v = (b_max[1] - b_min[1]) / (R - 1)
    dz_v = (b_max[2] - b_min[2]) / (R - 1)

    grad_x = np.gradient(sdf, dx_v, axis=0)
    grad_y = np.gradient(sdf, dy_v, axis=1)
    grad_z = np.gradient(sdf, dz_v, axis=2)
    normals = np.stack([grad_x, grad_y, grad_z], axis=0)

    sdf_vol = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0).to(cuda)
    normal_vol = torch.from_numpy(normals).float().unsqueeze(0).to(cuda)
    b_min_t = torch.tensor(b_min, dtype=torch.float32, device=cuda).unsqueeze(1)
    b_max_t = torch.tensor(b_max, dtype=torch.float32, device=cuda).unsqueeze(1)

    # ================================================================
    #  6. Load roots + synthesize strands via RK4
    # ================================================================
    print("\nStep 6: Loading roots + synthesizing strands...")
    from lib.hair_util import get_hair_root, save_strands_with_mesh
    from scripts.run_pde_generation import (
        hair_synthesis_rk4,
        project_points,
        query_2d_map,
        query_grid,
    )

    root_tensor = (
        torch.from_numpy(get_hair_root(args.roots))
        .float()
        .unsqueeze(0)
        .to(cuda)
    )
    calib_tensor = (
        torch.from_numpy(calibs["front"][0]).float().unsqueeze(0).to(cuda)
    )

    # Divergence map (from front strand_map, consistent with original PDE pipeline)
    # imageio loads RGB: index 0=R(mask), 1=G(dy), 2=B(dx).
    # Original uses cv2 BGR: index 0=B(dx), 1=G(dy).  Match that.
    # front_strand_bgr is cv2 BGR: index 0=B(dx), 1=G(dy), 2=R(mask)
    strand_img = front_strand_bgr
    strand_dx = -strand_img[:, :, 0]  # B channel → dx
    strand_dy = -strand_img[:, :, 1]  # G channel → -dy
    mask = (front_depth > 0.05).astype(np.float32)

    dx_dx = cv2.Sobel(strand_dx, cv2.CV_32F, 1, 0, ksize=3)
    dy_dy = cv2.Sobel(strand_dy, cv2.CV_32F, 0, 1, ksize=3)
    divergence = (dx_dx + dy_dy) * mask
    divergence = cv2.GaussianBlur(divergence, (5, 5), 0)

    div_map_t = (
        torch.from_numpy(divergence)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(cuda)
    )

    # Noise field
    import scipy.ndimage

    np.random.seed(42)
    raw_noise = np.random.randn(3, 64, 64, 64).astype(np.float32)
    smooth_noise = np.zeros_like(raw_noise)
    for c in range(3):
        smooth_noise[c] = scipy.ndimage.gaussian_filter(raw_noise[c], sigma=3.0)
    smooth_noise = smooth_noise / (np.std(smooth_noise) + 1e-8)
    noise_vol = torch.from_numpy(smooth_noise).unsqueeze(0).to(cuda)

    # Guide strands (KMeans)
    print("  Selecting 1024 guide strands via KMeans...")
    from sklearn.cluster import MiniBatchKMeans

    roots_3d = root_tensor.squeeze(0)
    N_roots = roots_3d.shape[1]
    pts_homo = torch.cat([roots_3d, torch.ones(1, N_roots, device=cuda)], dim=0)
    uv = torch.matmul(calib_tensor.squeeze(0), pts_homo)
    uv = uv[:2, :] / (uv[3:4, :] + 1e-8)
    uv_px = ((uv + 1.0) * 0.5 * 511).long().clamp(0, 511)

    uv_np = uv_px.float().cpu().numpy().T
    kmeans = MiniBatchKMeans(
        n_clusters=1024, random_state=42, n_init="auto", batch_size=2048
    ).fit(uv_np)
    centroids_list = kmeans.cluster_centers_.tolist()

    guide_idx_list = []
    for cx, cy in centroids_list:
        dist_sq = (uv_px[0].float() - cx) ** 2 + (uv_px[1].float() - cy) ** 2
        guide_idx_list.append(torch.argmin(dist_sq).item())

    guide_idx_tensor = torch.tensor(guide_idx_list, device=cuda)
    guide_roots = root_tensor[:, :, guide_idx_tensor]

    centroids_t = torch.tensor(centroids_list, device=cuda)

    # Nearest-centroid mapping for all roots
    dist_sq_all = (
        (uv_px[0].unsqueeze(0) - centroids_t[:, 0].unsqueeze(1)) ** 2
        + (uv_px[1].unsqueeze(0) - centroids_t[:, 1].unsqueeze(1)) ** 2
    )
    guide_indices_t = torch.argmin(dist_sq_all, dim=0)

    # Trace guide strands
    print("  Tracing 1024 guide strands...")
    guide_strands = hair_synthesis_rk4(
        strategy, cuda, guide_roots, calib_tensor,
        num_sample=args.num_sample, hair_unit=args.hair_unit,
        sdf_vol=sdf_vol, normal_vol=normal_vol,
        b_min_t=b_min_t, b_max_t=b_max_t,
    )
    guide_strands_t = (
        torch.from_numpy(guide_strands).to(cuda).permute(1, 2, 0)
    )

    # Trace all strands with guide clustering
    print(f"  Tracing {N_roots} full strands...")
    valid_cluster_mask = (
        torch.from_numpy(mask).to(cuda)[uv_px[1], uv_px[0]]
    )

    strands = hair_synthesis_rk4(
        strategy, cuda, root_tensor, calib_tensor,
        num_sample=args.num_sample, hair_unit=args.hair_unit,
        sdf_vol=sdf_vol, normal_vol=normal_vol,
        b_min_t=b_min_t, b_max_t=b_max_t,
        noise_vol=noise_vol,
        guide_strands=guide_strands_t,
        guide_indices=guide_indices_t,
        div_map_t=div_map_t,
        valid_cluster_mask=valid_cluster_mask,
    )

    # ================================================================
    #  7. Save
    # ================================================================
    print(f"\nStep 7: Clipping & saving to {args.out_ply}...")
    save_strands_with_mesh(strands, args.mesh_obj, args.out_ply, 0.3, is_eval=False)
    print(f"Done → {args.out_ply}")


if __name__ == "__main__":
    main()
