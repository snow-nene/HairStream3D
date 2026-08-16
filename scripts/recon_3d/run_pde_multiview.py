"""
Multi-View 3D Hair Synthesis via PDE.

Loads 4-view strand_maps + depth_maps, fuses them into a 3D orientation
volume (front=exclusive ground truth), solves Laplace PDE, and traces
strands via RK4 integration.

Usage:
  python scripts/recon_3d/run_pde_multiview.py \
      --strand_dir  results/multiview_strand_depth/strand_map \
      --depth_dir   results/multiview_strand_depth/depth_map \
      --mesh_obj    results/test_pixal3d/hair_mesh_flame_extracted.obj \
      --out_ply     results/multiview_pde/hair.ply
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import argparse
import numpy as np
import torch
import cv2
import trimesh
import open3d as o3d
import imageio.v2 as imageio

from lib.multiview_fusion import (
    align_vector_sign,
    build_mesh_root_guidance,
    build_mesh_metric_band,
    build_blender_calib,
    build_multiview_seg_support_volume,
    build_visible_mesh_surface_shell,
    compute_root_head_visibility,
    extract_view_mesh_surface_contribution,
    fuse_multiview_orientation,
    orient_sparse_direction_axes,
    save_mesh_root_projections,
    trace_view_strands_3d,
)
from lib.multiview_pde import MultiViewLaplacePDEStrategy


def query_grid(vol_5d, points_3n, b_min_tensor, b_max_tensor):
    """Sample a 3D volume at world-space points."""
    uv = (points_3n.unsqueeze(0) - b_min_tensor) / (
        b_max_tensor - b_min_tensor
    )
    uv = uv * 2.0 - 1.0
    grid = uv.permute(0, 2, 1)
    grid = grid[..., [2, 1, 0]].unsqueeze(2).unsqueeze(2)
    value = torch.nn.functional.grid_sample(
        vol_5d,
        grid,
        padding_mode="border",
        align_corners=True,
    )
    return value.squeeze(-1).squeeze(-1).squeeze(0)


def project_points(points_3d, calib_tensor):
    """Project 3D points to image NDC coordinates."""
    num_points = points_3d.shape[1]
    homogeneous = torch.cat(
        [
            points_3d,
            torch.ones(1, num_points, device=points_3d.device),
        ],
        dim=0,
    )
    uv = torch.matmul(calib_tensor.squeeze(0), homogeneous)
    return uv[:2, :] / (uv[3:4, :] + 1e-8)


def query_2d_map(map_tensor, uv):
    """Sample an image-space tensor at NDC coordinates."""
    grid = uv.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1)
    value = torch.nn.functional.grid_sample(
        map_tensor,
        grid,
        padding_mode="border",
        align_corners=True,
    )
    return value.squeeze(-1).squeeze(0)


def hair_synthesis_rk4(
    strategy,
    cuda,
    root_tensor,
    calib_tensor,
    num_sample=100,
    hair_unit=0.006,
    sdf_vol=None,
    normal_vol=None,
    b_min_t=None,
    b_max_t=None,
    noise_vol=None,
    guide_strands=None,
    guide_indices=None,
    div_map_t=None,
    valid_cluster_mask=None,
    fallback_steps=12,
    silhouette_guard=None,
    silhouette_grace_steps=3,
    max_turn_degrees=0.0,
    actual_displacement_feedback=False,
    label="",
):
    """Trace strands with RK4 integration and continuous collision response.

    silhouette_guard (SilhouetteGuard): 可见性驱动的轮廓硬约束。每步把候选
    点投影回输入视角：明显越过 seg 的发丝立即停止（冻结在最后合法位置），
    刚越界 (<= tolerance_px) 的发丝进入 grace 期并允许 fallback 方向帮助其
    回到轮廓内部。
    """
    from lib.silhouette_guard import STATUS_HARD, STATUS_SOFT

    num_strands = root_tensor.shape[2]
    hair_strands = torch.zeros(
        num_sample, 3, num_strands, device=cuda
    )
    current = root_tensor.squeeze(0)
    hair_strands[0] = current
    previous_direction = None

    alive = torch.ones(num_strands, dtype=torch.bool, device=cuda)
    grace_budget = max(0, int(silhouette_grace_steps))
    grace_left = torch.full(
        (num_strands,), grace_budget, dtype=torch.long, device=cuda
    )
    need_fallback = torch.zeros(num_strands, dtype=torch.bool, device=cuda)
    dead_total = 0

    def normalize_nonzero(vectors):
        norms = torch.norm(vectors, dim=0, keepdim=True)
        return torch.where(
            norms > 1e-8,
            vectors / torch.clamp(norms, min=1e-8),
            vectors,
        )

    def query_direction(points, fallback_mask):
        direction = strategy.query(points, calib_tensor).squeeze(0)
        low = torch.norm(direction, dim=0) < 0.05
        use_fallback = low & fallback_mask
        if use_fallback.any() and hasattr(strategy, "query_fallback"):
            fallback = strategy.query_fallback(points, calib_tensor).squeeze(0)
            direction = torch.where(use_fallback.unsqueeze(0), fallback, direction)
        return direction

    for index in range(1, num_sample):
        step_origin = current
        fallback_mask = (index <= int(fallback_steps)) | need_fallback
        k1 = query_direction(current.unsqueeze(0), fallback_mask)
        if previous_direction is not None:
            k1 = align_vector_sign(k1, previous_direction)
        k2 = query_direction(
            (current + 0.5 * hair_unit * k1).unsqueeze(0), fallback_mask
        )
        k2 = align_vector_sign(k2, k1)
        k3 = query_direction(
            (current + 0.5 * hair_unit * k2).unsqueeze(0), fallback_mask
        )
        k3 = align_vector_sign(k3, k2)
        k4 = query_direction(
            (current + hair_unit * k3).unsqueeze(0), fallback_mask
        )
        k4 = align_vector_sign(k4, k3)
        direction = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if previous_direction is not None:
            direction = align_vector_sign(direction, previous_direction)
        progress = index / float(num_sample)

        if div_map_t is not None:
            uv = project_points(current, calib_tensor)
            divergence = query_2d_map(div_map_t, uv).squeeze(0)
        else:
            divergence = torch.zeros(num_strands, device=cuda)

        magnitude = torch.norm(direction, dim=0, keepdim=True)
        if noise_vol is not None:
            noise = query_grid(
                noise_vol, current, b_min_t, b_max_t
            )
            is_divergent = (divergence > 0.1).float()
            divergence_magnitude = torch.clamp(divergence, 0.0, 1.0)
            noise_weight = (
                0.5
                * progress**2
                * divergence_magnitude
                * is_divergent
            )
            direction = (
                direction
                + noise_weight.unsqueeze(0) * noise * magnitude
            )
        if previous_direction is not None and max_turn_degrees > 0.0:
            previous_unit = normalize_nonzero(previous_direction)
            direction_norm = torch.norm(direction, dim=0, keepdim=True)
            direction_unit = normalize_nonzero(direction)
            cosine = torch.clamp(
                torch.sum(previous_unit * direction_unit, dim=0),
                -1.0,
                1.0,
            )
            angle = torch.acos(cosine)
            max_angle = torch.deg2rad(
                torch.tensor(max_turn_degrees, device=cuda)
            )
            blend = torch.clamp(max_angle / torch.clamp(angle, min=1e-6), max=1.0)
            limited_unit = normalize_nonzero(
                torch.lerp(previous_unit, direction_unit, blend.unsqueeze(0))
            )
            limited = angle > max_angle
            direction = torch.where(
                limited.unsqueeze(0),
                limited_unit * direction_norm,
                direction,
            )

        if not actual_displacement_feedback:
            previous_direction = direction

        current = current + hair_unit * direction

        if guide_strands is not None and guide_indices is not None:
            guide_points = guide_strands[index, :, guide_indices]
            is_clumping = (divergence < -0.1).float()
            clump_magnitude = torch.clamp(-divergence, 0.0, 1.0)
            ramp = min(1.0, progress / 0.5)
            clump_weight = torch.clamp(
                0.005 * ramp
                + 0.01 * clump_magnitude * is_clumping * ramp,
                0.0,
                0.02,
            )
            if valid_cluster_mask is not None:
                clump_weight = clump_weight * valid_cluster_mask
            magnitude_weight = torch.clamp(
                magnitude / 0.9, 0.0, 1.0
            ).squeeze(0)
            clump_weight = clump_weight * magnitude_weight
            current = torch.lerp(
                current, guide_points, clump_weight.unsqueeze(0)
            )

        if sdf_vol is not None:
            sdf = query_grid(sdf_vol, current, b_min_t, b_max_t)
            normal = query_grid(normal_vol, current, b_min_t, b_max_t)
            normal = torch.nn.functional.normalize(normal, dim=0)
            penetration = 0.005 - sdf
            collision_mask = (penetration > 0).float()
            current = current + collision_mask * penetration * normal

        if silhouette_guard is not None and alive.any():
            # 可见性驱动的 silhouette 硬约束：只检查仍存活的发丝
            if alive.all():
                check_idx = None
                check_pts = current
            else:
                check_idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
                check_pts = current[:, check_idx]
            status_np = silhouette_guard.classify(
                check_pts.detach().cpu().numpy().T
            )
            status_t = torch.from_numpy(status_np.astype(np.int64)).to(cuda)
            hard = status_t == STATUS_HARD
            soft = status_t == STATUS_SOFT
            if check_idx is None:
                grace_left = torch.where(
                    soft, grace_left - 1, torch.full_like(grace_left, grace_budget)
                )
                newly_dead = hard | (soft & (grace_left < 0))
                alive = alive & ~newly_dead
                need_fallback = alive & soft
                dead_step = int(newly_dead.sum())
                current = torch.where(alive.unsqueeze(0), current, hair_strands[index - 1])
            else:
                sub_grace = grace_left[check_idx]
                sub_grace = torch.where(
                    soft, sub_grace - 1, torch.full_like(sub_grace, grace_budget)
                )
                grace_left[check_idx] = sub_grace
                newly_dead_sub = hard | (soft & (sub_grace < 0))
                dead_idx = check_idx[newly_dead_sub]
                alive[dead_idx] = False
                need_fallback = torch.zeros_like(need_fallback)
                need_fallback[check_idx[soft & ~newly_dead_sub]] = True
                dead_step = int(newly_dead_sub.sum())
                if dead_step:
                    current[:, dead_idx] = hair_strands[index - 1, :, dead_idx]
            dead_total += dead_step
            if index % 25 == 0 or index == num_sample - 1:
                print(
                    f"    [{label or 'rk4'}] step {index}: "
                    f"alive={int(alive.sum())}/{num_strands}, "
                    f"stopped so far={dead_total}"
                )

        # Dead strands must remain at their last valid point. Without this
        # unconditional mask, later clumping/collision updates can move a strand
        # again after the silhouette or domain guard has stopped it.
        current = torch.where(
            alive.unsqueeze(0), current, hair_strands[index - 1]
        )

        if actual_displacement_feedback:
            actual_direction = (current - step_origin) / max(hair_unit, 1e-8)
            actual_valid = torch.norm(actual_direction, dim=0) > 1e-8
            previous_direction = torch.where(
                actual_valid.unsqueeze(0), actual_direction, direction
            )

        hair_strands[index] = current

    if silhouette_guard is not None:
        print(
            f"  [{label or 'rk4'}] silhouette guard: stopped "
            f"{dead_total}/{num_strands} strands outside the input silhouette"
        )

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def smooth_strands_laplacian(strands, iterations=0, strength=0.5):
    """Smooth moving strand interiors while preserving roots and endpoints."""
    if iterations <= 0 or strength <= 0.0:
        return strands

    smoothed = np.asarray(strands, dtype=np.float32).copy()
    segment_lengths = np.linalg.norm(
        smoothed[:, 1:] - smoothed[:, :-1], axis=2
    )
    effective_points = 1 + (segment_lengths > 1e-7).sum(axis=1)
    sample_ids = np.arange(1, smoothed.shape[1] - 1)[None, :]
    movable = sample_ids < (effective_points - 1)[:, None]
    weight = float(np.clip(strength, 0.0, 1.0))

    for _ in range(int(iterations)):
        midpoint = 0.5 * (smoothed[:, :-2] + smoothed[:, 2:])
        updated = (1.0 - weight) * smoothed[:, 1:-1] + weight * midpoint
        smoothed[:, 1:-1] = np.where(
            movable[:, :, None], updated, smoothed[:, 1:-1]
        )

    return smoothed


def load_blender_view_calibration(data_dir, view):
    """把 Blender 真值相机转换为 HairStep-world → image-NDC 标定。

    Blender 渲染使用原始 GLB，而 PDE 体素位于 ``glb_to_world.npz`` 定义的
    HairStep 世界坐标系。完整变换为：

        HairStep world → raw glTF → Blender world → camera clip

    返回的标定矩阵预先翻转 NDC y，使其兼容融合代码的左上图像原点；纯
    旋转矩阵仍使用 Blender camera 的 y-up 坐标，供 2D strand 方向反投影。
    """
    camera_path = os.path.join(
        data_dir, "blender_renders", "camera_params", f"{view}.npz"
    )
    transform_path = os.path.join(data_dir, "pixal3d", "glb_to_world.npz")
    if not os.path.exists(camera_path):
        raise FileNotFoundError(
            f"缺少 {view} 真值相机: {camera_path}。请重新运行 render 阶段，"
            "或用 render_multiview_blender.py --camera_only 补齐。"
        )
    if not os.path.exists(transform_path):
        raise FileNotFoundError(
            f"缺少 GLB→HairStep 变换: {transform_path}。请先运行 extract 阶段。"
        )

    camera = np.load(camera_path)
    transform = np.load(transform_path)

    umeyama = np.eye(4, dtype=np.float64)
    umeyama[:3, :3] = (
        float(np.asarray(transform["umeyama_scale"]).reshape(-1)[0])
        * np.asarray(transform["umeyama_R"], dtype=np.float64)
    )
    umeyama[:3, 3] = np.asarray(transform["umeyama_t"], dtype=np.float64)
    gltf_to_hairstep = np.asarray(transform["icp_T"], dtype=np.float64) @ umeyama

    # glTF is Y-up; Blender imports it as Z-up: (x, y, z) → (x, -z, y).
    gltf_to_blender = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    hairstep_to_blender = gltf_to_blender @ np.linalg.inv(gltf_to_hairstep)
    world_to_clip = np.asarray(camera["world_to_clip"], dtype=np.float64)

    # Existing fusion maps NDC y directly to image rows, so convert y-up to y-down.
    ndc_to_image = np.diag([1.0, -1.0, 1.0, 1.0])
    calib = ndc_to_image @ world_to_clip @ hairstep_to_blender

    # Neural depth maps are normalized to [0, 1] per view, whereas Blender's
    # OpenGL clip-z is near -1 for this orthographic scene. Normalize the exact
    # camera-space depth over the rendered mesh range; larger camera-z is nearer.
    if "camera_depth_min" not in camera or "camera_depth_max" not in camera:
        raise KeyError(
            f"{camera_path} 缺少 camera_depth_min/max，请用新版渲染脚本重新导出。"
        )
    hair_to_camera = (
        np.asarray(camera["world_to_camera"], dtype=np.float64)
        @ hairstep_to_blender
    )
    depth_min = float(camera["camera_depth_min"])
    depth_max = float(camera["camera_depth_max"])
    calib[2] = (
        hair_to_camera[2] - depth_min * hair_to_camera[3]
    ) / max(depth_max - depth_min, 1e-8)

    # Direction back-projection needs rotation only (no Umeyama scale/translation).
    gltf_to_hairstep_rotation = (
        np.asarray(transform["icp_T"], dtype=np.float64)[:3, :3]
        @ np.asarray(transform["umeyama_R"], dtype=np.float64)
    )
    camera_rotation = np.asarray(camera["world_to_camera"], dtype=np.float64)[:3, :3]
    rotation = (
        camera_rotation
        @ gltf_to_blender[:3, :3]
        @ gltf_to_hairstep_rotation.T
    )
    # Remove small numerical drift while retaining a proper SO(3) rotation.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    return calib.astype(np.float32), rotation.astype(np.float32)


def resolve_front_calibration_paths(
    data_dir,
    front_calib_override=None,
    front_depth_calib_override=None,
):
    """选择 front 姿态标定，并为稠密标定保留旧神经深度投影行。"""
    param_dir = os.path.join(data_dir, "maps", "param")
    legacy_path = os.path.join(param_dir, "front.npy")
    dense_path = os.path.join(param_dir, "front_dense_silhouette.npy")
    if front_calib_override:
        return front_calib_override, front_depth_calib_override
    if os.path.exists(dense_path):
        depth_path = front_depth_calib_override or legacy_path
        return dense_path, depth_path
    return legacy_path, front_depth_calib_override


def main():
    parser = argparse.ArgumentParser(
        description="Multi-View 3D Hair PDE Synthesis"
    )
    parser.add_argument("--img_id", required=True,
                        help="Image ID used for multiview_data directory")
    parser.add_argument("--views", nargs="*", default=["front"],
                        help="List of views to use (default: front)")
    parser.add_argument("--expand-multiview", action="store_true",
                        help="If set, automatically load all available views (front, left, right, back)")
    parser.add_argument("--mesh_obj",
                        default=None,
                        help="Mesh object to use for bounds (default fallback)")
    parser.add_argument(
        "--front_calib",
        default=None,
        help="Optional front calibration override; defaults to maps/param/front.npy",
    )
    parser.add_argument(
        "--front_depth_calib",
        default=None,
        help=(
            "Optional calibration whose third projection row supplies the "
            "front neural-depth scale while --front_calib supplies x/y pose"
        ),
    )
    parser.add_argument("--out_dir",
                        default=None,
                        help="Output directory (default: results/multiview_data/<img_id>/pde_reconstruction)")
    parser.add_argument("--roots",
                        default="data/roots10k.obj")
    parser.add_argument("--pde_resolution", type=int, default=384)
    parser.add_argument("--pde_dilation_iters", type=int, default=25)
    parser.add_argument(
        "--pde_band_width",
        type=float,
        default=0.05,
        help="Metric hair-mesh PDE band width in meters (default: 0.05)",
    )
    parser.add_argument(
        "--head_occlusion_tolerance",
        type=float,
        default=0.03,
        help="Head/hair alignment tolerance for visibility tests in meters",
    )
    parser.add_argument("--pde_cg_tol", type=float, default=1e-4)
    parser.add_argument("--pde_cg_maxiter", type=int, default=2000)
    parser.add_argument("--pde_anisotropy", type=float, default=0.8)
    parser.add_argument(
        "--pde_max_normal_component",
        type=float,
        default=0.30,
        help="Maximum absolute hair-mesh normal component in the multi-view field",
    )
    parser.add_argument(
        "--pde_harmonic_relax_iters",
        type=int,
        default=8,
        help="Number of clamped Gaussian harmonic-relaxation iterations",
    )
    parser.add_argument(
        "--pde_scalp_boundary_width",
        type=float,
        default=0.005,
        help="Width in meters of the artificial scalp boundary (default: 0.005)",
    )
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
    parser.add_argument(
        "--pde_fallback_steps",
        type=int,
        default=12,
        help="Early RK4 steps allowed to use nearest-valid PDE direction",
    )
    parser.add_argument(
        "--rk4_max_turn_degrees",
        type=float,
        default=0.0,
        help="Optional maximum direction turn per RK4 step; <=0 disables it",
    )
    parser.add_argument(
        "--rk4_actual_displacement_feedback",
        action="store_true",
        help=(
            "Use the post-clump/post-collision displacement as the previous "
            "RK4 direction on the next step"
        ),
    )
    parser.add_argument(
        "--strand_smoothing_iters",
        type=int,
        default=0,
        help="Laplacian smoothing passes applied before final collision/trim",
    )
    parser.add_argument(
        "--strand_smoothing_strength",
        type=float,
        default=0.5,
        help="Per-pass Laplacian smoothing strength in [0, 1]",
    )
    parser.add_argument(
        "--silhouette_hard_px",
        type=float,
        default=20.0,
        help="Immediate-stop depth outside the input hair seg in pixels "
             "(shallower excursions use the grace mechanism, default: 20.0)",
    )
    parser.add_argument(
        "--silhouette_grace_steps",
        type=int,
        default=12,
        help="Max consecutive RK4 steps a strand may stay outside the seg "
             "(grace for transient hairline excursions, default: 12)",
    )
    parser.add_argument(
        "--silhouette_trim_margin_px",
        type=float,
        default=1.0,
        help="Save-stage silhouette trim margin in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--disable_silhouette_guard",
        action="store_true",
        help="Disable the visibility-driven silhouette constraint (ablation)",
    )
    parser.add_argument(
        "--silhouette_multiview_union",
        action="store_true",
        help=(
            "Allow any visible view's hair seg to support a point instead of "
            "letting the primary/front view veto it (ablation)"
        ),
    )
    parser.add_argument(
        "--export-per-view",
        action="store_true",
        help=(
            "Trace each view's 2D strand map and back-project it with that view's "
            "depth map under pde_reconstruction/per_view/<view>/"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="PyTorch execution device; auto falls back to CPU when CUDA is unavailable",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device_name = (
        "cuda:0"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    cuda = torch.device(device_name)
    print(f"Execution device: {cuda}")

    data_dir = os.path.join("results", "multiview_data", args.img_id)
    strand_dir = os.path.join(data_dir, "maps", "strand_map")
    depth_dir = os.path.join(data_dir, "maps", "depth_map")
    
    out_dir = args.out_dir if args.out_dir else os.path.join(data_dir, "pde_reconstruction")
    out_ply = os.path.join(out_dir, "hair_multiview.ply")
    os.makedirs(out_dir, exist_ok=True)

    # ================================================================
    #  1. Load multi-view strand_maps + depth_maps
    # ================================================================
    print("=" * 60)
    print("Step 1: Loading multi-view data...")
    
    if args.expand_multiview:
        target_views = ["front", "left", "right", "back"]
    else:
        target_views = args.views

    strand_maps = {}
    depth_maps = {}
    valid_views = []

    for v in target_views:
        sp = os.path.join(strand_dir, f"{v}.png")
        dp = os.path.join(depth_dir, f"{v}.npy")
        
        if not os.path.exists(sp) or not os.path.exists(dp):
            print(f"  Warning: Missing strand or depth map for view '{v}'. Skipping.")
            continue
            
        strand_maps[v] = imageio.imread(sp).astype(np.float32) / 255.0
        depth_maps[v] = np.load(dp).astype(np.float32)
        print(f"  {v}: strand={strand_maps[v].shape}, depth={depth_maps[v].shape}")
        valid_views.append(v)
        
    if len(valid_views) == 0:
        raise RuntimeError("No valid views found for PDE reconstruction.")

    # `seg` is the authoritative hair mask. The strand-map mask channel can
    # include face/background pixels and must not gate mesh/root guidance.
    seg_masks = {}
    for view in valid_views:
        seg_path = os.path.join(data_dir, "maps", "seg", f"{view}.png")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing hair seg: {seg_path}")
        seg = imageio.imread(seg_path)
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        if seg.shape != depth_maps[view].shape:
            seg = cv2.resize(
                seg.astype(np.uint8),
                (depth_maps[view].shape[1], depth_maps[view].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        seg_masks[view] = seg > 127

    # ================================================================
    #  2. Build camera calibration matrices
    # ================================================================
    print("\nStep 2: Building camera calibrations...")

    b_min, b_max = np.array([1e5] * 3), np.array([-1e5] * 3)
    
    # ALWAYS include head_model.obj in bounding box (so roots in the back don't get clipped)
    head_mesh_bbox = o3d.io.read_triangle_mesh("data/head_model.obj").get_axis_aligned_bounding_box()
    b_min = np.minimum(b_min, head_mesh_bbox.get_min_bound())
    b_max = np.maximum(b_max, head_mesh_bbox.get_max_bound())
    
    if args.mesh_obj is not None and os.path.exists(args.mesh_obj):
        mesh = o3d.io.read_triangle_mesh(args.mesh_obj)
        verts = np.asarray(mesh.vertices)
        if len(verts) > 0:
            extent = (verts.max(axis=0) - verts.min(axis=0)).max()
        else:
            print(f"[警告] {args.mesh_obj} 无法加载，退化为默认头部包围盒计算。")
            extent = 0.3 # 默认尺寸
            mesh = None
    else:
        print(f"[警告] 未找到 {args.mesh_obj}，退化为默认头部包围盒计算。")
        extent = 0.3
        mesh = None
    print(f"  Mesh: {len(verts) if mesh else 0} verts, extent={extent:.4f}")

    # Prefer the render-to-photo dense SO(3) + hair-silhouette calibration when
    # available. Its x/y rows align the Pixal3D geometry to the real photo, while
    # the legacy front.npy depth row stays coupled to the neural depth map.
    from scripts.recon_3d.recon3D import load_calib
    real_calib_path, front_depth_calib_path = resolve_front_calibration_paths(
        data_dir,
        front_calib_override=args.front_calib,
        front_depth_calib_override=args.front_depth_calib,
    )
    glb_calib_path = os.path.join(data_dir, "maps", "param", "glb_param.npy")

    if os.path.exists(real_calib_path):
        front_calib_path = real_calib_path
        print(f"  [Front Baseline] Using real front calibration: {front_calib_path}")
    elif os.path.exists(glb_calib_path):
        front_calib_path = glb_calib_path
        print(f"  [Fallback] Using GLB front calibration: {front_calib_path}")
    else:
        front_calib_path = None

    if front_calib_path is None or not os.path.exists(front_calib_path):
        print(f"  [WARN] Missing front camera calibration: {front_calib_path}")
        print(f"  [WARN] Falling back to dynamic front calibration.")
        
        import cv2
        img = cv2.imread(os.path.join(strand_dir, "front.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imread(os.path.join(data_dir, "maps", "strand", "front.png"), cv2.IMREAD_UNCHANGED)
            
        if img is not None:
            mask = img[:, :, 2] > 0
            y, x = np.where(mask)
            if len(y) > 0:
                px_min, px_max = x.min(), x.max()
                py_min, py_max = y.min(), y.max()
                
                head_mesh_tmp = o3d.io.read_triangle_mesh(args.mesh_obj)
                bbox = head_mesh_tmp.get_axis_aligned_bounding_box()
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                
                X_min, X_max = min_bound[0], max_bound[0]
                Y_min, Y_max = min_bound[1], max_bound[1]
                Z_min, Z_max = min_bound[2], max_bound[2]
                
                W = img.shape[1]
                H = img.shape[0]
                
                ortho_x = (X_max - X_min) / (px_max - px_min) * (W / 2)
                ortho_y = (Y_max - Y_min) / (py_max - py_min) * (H / 2)
                ortho_ratio = (ortho_x + ortho_y) / 2
                
                center_x = (X_max + X_min) / 2 - ((px_max + px_min) / (W / 2) - 2.0) * ortho_ratio / 2
                center_y = Y_max + (py_min / (H/2) - 1.0) * ortho_ratio
                center_z = (Z_max + Z_min) / 2
                
                b_center_tmp = np.array([center_x, center_y, center_z], dtype=np.float32)
                extent = ortho_ratio / 1.4
                print(f"  [INFO] Computed dynamic extent={extent:.4f}, center={b_center_tmp}")
            else:
                head_mesh_tmp = o3d.io.read_triangle_mesh(args.mesh_obj)
                b_center_tmp = head_mesh_tmp.get_axis_aligned_bounding_box().get_center()
                extent = 0.3
        else:
            head_mesh_tmp = o3d.io.read_triangle_mesh(args.mesh_obj)
            b_center_tmp = head_mesh_tmp.get_axis_aligned_bounding_box().get_center()
            extent = 0.3

        param = {
            'center': b_center_tmp.reshape(3, 1).astype(np.float32),
            'R': np.eye(3, dtype=np.float32),
            'scale': 1.0,
            'ortho_ratio': 1.0
        }
        front_calib, _ = build_blender_calib("front", b_center_tmp, extent)
    else:
        print(f"  Loading REAL front calib from: {front_calib_path}")
        param = np.load(front_calib_path, allow_pickle=True).item()
        if 'ortho_ratio' in param:
            extent = param['ortho_ratio'] / 1.4
            print(f"  Updated extent from real ortho_ratio: {extent:.4f}")
        front_calib = load_calib(front_calib_path, loadSize=1024)
        if isinstance(front_calib, torch.Tensor):
            front_calib = front_calib.numpy()
        if front_depth_calib_path:
            depth_calib = load_calib(front_depth_calib_path, loadSize=1024)
            if isinstance(depth_calib, torch.Tensor):
                depth_calib = depth_calib.numpy()
            front_calib[2] = depth_calib[2]
            print(
                "  Front depth row preserved from: "
                f"{front_depth_calib_path}"
            )

    real_center = param.get('center').flatten().astype(np.float32)
    R_real = param.get('R').astype(np.float32)
    R_real = R_real / (np.linalg.norm(R_real, axis=1, keepdims=True) + 1e-8)
    
    geometry_calibs = {"front": (front_calib.astype(np.float32), R_real)}
    print(f"  Real center: {real_center}")

    # Side/back: use the exact cameras that produced blender_renders/<view>.png.
    for v in valid_views:
        if v == "front":
            continue
        geometry_calibs[v] = load_blender_view_calibration(data_dir, v)
        print(f"  {v}: loaded Blender ground-truth camera")

    mesh_root_guidance = None
    if any(v != "front" for v in valid_views):
        guidance_mesh_path = os.path.join(data_dir, "pixal3d", "hair_mesh_sdf.obj")
        if not os.path.exists(guidance_mesh_path):
            guidance_mesh_path = os.path.join(
                data_dir, "pixal3d", "hair_mesh_aligned_best.obj"
            )
        if os.path.exists(guidance_mesh_path):
            from lib.hair_util import get_hair_root

            roots_world = get_hair_root(args.roots).T
            mesh_root_guidance = build_mesh_root_guidance(
                guidance_mesh_path,
                roots_world,
                geometry_calibs,
                strand_maps,
                seg_masks=seg_masks,
                head_mesh_path="data/head_model.obj",
            )
            guidance_output = {
                "roots_world": roots_world.astype(np.float32),
                "mesh_path": np.asarray(str(guidance_mesh_path)),
            }
            for view, visible in mesh_root_guidance["visible_roots"].items():
                guidance_output[f"{view}_visible_roots"] = visible
            np.savez_compressed(
                os.path.join(out_dir, "root_view_guidance.npz"),
                **guidance_output,
            )
            background_paths = {
                view: os.path.join(data_dir, "blender_renders", f"{view}.png")
                for view in valid_views
            }
            raw_path_file = os.path.join(data_dir, "raw_img_path.txt")
            if os.path.exists(raw_path_file):
                with open(raw_path_file, "r", encoding="utf-8") as file:
                    background_paths["front"] = file.read().strip()
            save_mesh_root_projections(
                mesh_root_guidance,
                geometry_calibs,
                strand_maps,
                background_paths,
                os.path.join(out_dir, "mesh_root_projection"),
                seg_masks=seg_masks,
            )
            print(f"  Mesh/root guidance: {guidance_mesh_path}")
        else:
            print("  [WARN] Hair mesh missing; side views will use legacy depth lifting")

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
    
    if mesh is not None:
        hair_bbox = mesh.get_axis_aligned_bounding_box()
        b_min = np.minimum(b_min, hair_bbox.get_min_bound())
        b_max = np.maximum(b_max, hair_bbox.get_max_bound())
        padding = 0.03
    else:
        # 如果没有发型网格，提供一个更大的容差，假设头发最高可达 0.12 的偏移
        padding = 0.10
        
    b_min = b_min - padding
    b_max = b_max + padding
    print(f"  BBox: [{b_min}, {b_max}]")

    fused_orien, boundary_mask, view_ownership = fuse_multiview_orientation(
        strand_maps=strand_maps,
        depth_maps=depth_maps,
        calibs=geometry_calibs,
        resolution=args.pde_resolution,
        b_min=b_min,
        b_max=b_max,
        center=real_center,
        extent=extent,
        front_surface_margin=args.front_surface_margin,
        other_surface_margin=args.other_surface_margin,
        mesh_root_guidance=mesh_root_guidance,
    )

    # Build the PDE domain from first-visible, hair-segmented hits on the
    # aligned Pixal3D mesh.  Do not use the bald-head SDF to trim this shell.
    aligned_surface_path = os.path.join(
        data_dir, "pixal3d", "hair_mesh_aligned_best.obj"
    )
    if not os.path.exists(aligned_surface_path):
        aligned_surface_path = args.mesh_obj
    if not aligned_surface_path or not os.path.exists(aligned_surface_path):
        raise FileNotFoundError(
            "Visible surface shell requires pixal3d/hair_mesh_aligned_best.obj "
            "or --mesh_obj"
        )
    silhouette_guard = None
    if not args.disable_silhouette_guard:
        from lib.silhouette_guard import SilhouetteGuard

        primary_view = "front" if "front" in seg_masks else valid_views[0]
        silhouette_guard = SilhouetteGuard(
            seg_masks=seg_masks,
            calibs=geometry_calibs,
            head_mesh_path="data/head_model.obj",
            hard_px=args.silhouette_hard_px,
            primary_view=primary_view,
            primary_authoritative=not args.silhouette_multiview_union,
        )
        print(
            f"  {silhouette_guard.summary()}: RK4 growth is silhouette-constrained "
            f"(grace={args.silhouette_grace_steps} steps)"
        )
    else:
        print("  Silhouette guard DISABLED (--disable_silhouette_guard)")

    hair_volume, raw_surface_shell, shell_counts = build_visible_mesh_surface_shell(
        aligned_surface_path,
        geometry_calibs,
        seg_masks,
        b_min,
        b_max,
        args.pde_resolution,
        shell_iterations=2,
    )
    # Replace the legacy depth-surface boundary with the actual per-view mesh
    # surface contributions. Fuse the signless directions through their
    # second-moment tensor, then use its principal eigenvector per voxel.
    contribution_records = {}
    flat_ids_all = []
    directions_all = []
    view_ids_all = []
    shape = np.asarray(hair_volume.shape, dtype=np.int64)
    for view_id, view in enumerate(valid_views):
        contribution_points, contribution_lines = (
            extract_view_mesh_surface_contribution(
                aligned_surface_path,
                strand_maps[view],
                seg_masks[view],
                geometry_calibs[view],
            )
        )
        contribution_records[view] = (
            contribution_points,
            contribution_lines,
        )
        count = len(contribution_lines)
        starts = contribution_points[:count]
        directions = contribution_points[count:] - starts
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        grid = np.rint(
            (starts - b_min) / np.maximum(b_max - b_min, 1e-12)
            * (shape - 1)
        ).astype(np.int64)
        inside = np.all((grid >= 0) & (grid < shape), axis=1)
        grid = grid[inside]
        flat_ids_all.append(np.ravel_multi_index(grid.T, tuple(shape)))
        directions_all.append(directions[inside])
        view_ids_all.append(np.full(inside.sum(), view_id, dtype=np.int8))

    flat_ids = np.concatenate(flat_ids_all)
    directions = np.concatenate(directions_all)
    contributing_views = np.concatenate(view_ids_all)
    order = np.argsort(flat_ids)
    flat_ids = flat_ids[order]
    directions = directions[order]
    contributing_views = contributing_views[order]
    unique_ids, starts = np.unique(flat_ids, return_index=True)
    tensor_terms = np.column_stack([
        directions[:, 0] * directions[:, 0],
        directions[:, 0] * directions[:, 1],
        directions[:, 0] * directions[:, 2],
        directions[:, 1] * directions[:, 1],
        directions[:, 1] * directions[:, 2],
        directions[:, 2] * directions[:, 2],
    ])
    moments = np.add.reduceat(tensor_terms, starts, axis=0)
    tensors = np.empty((len(unique_ids), 3, 3), dtype=np.float32)
    tensors[:, 0, 0] = moments[:, 0]
    tensors[:, 0, 1] = tensors[:, 1, 0] = moments[:, 1]
    tensors[:, 0, 2] = tensors[:, 2, 0] = moments[:, 2]
    tensors[:, 1, 1] = moments[:, 3]
    tensors[:, 1, 2] = tensors[:, 2, 1] = moments[:, 4]
    tensors[:, 2, 2] = moments[:, 5]
    _, eigenvectors = np.linalg.eigh(tensors)
    fused_directions = eigenvectors[:, :, -1]
    reference = directions[starts]
    flip = np.sum(fused_directions * reference, axis=1) < 0.0
    fused_directions[flip] *= -1.0
    fused_ownership = np.minimum.reduceat(contributing_views, starts)
    gravity_oriented_owners = [
        view_id for view_id, view in enumerate(valid_views)
        if view != "front"
    ]
    fused_directions = orient_sparse_direction_axes(
        unique_ids,
        fused_directions,
        reference,
        fused_ownership,
        shape,
        preferred_owners=gravity_oriented_owners,
    )
    for view_id, view in enumerate(valid_views):
        owned = fused_ownership == view_id
        if owned.any():
            downward = np.mean(fused_directions[owned, 1] < 0.0)
            print(
                f"  {view} oriented boundary: "
                f"{100.0 * downward:.1f}% world-down"
            )

    fused_orien = np.zeros_like(fused_orien)
    boundary_mask = np.zeros_like(boundary_mask)
    view_ownership = np.full_like(view_ownership, -1)
    fused_flat = fused_orien.reshape(3, -1)
    fused_flat[:, unique_ids] = fused_directions.T
    boundary_mask.ravel()[unique_ids] = True
    view_ownership.ravel()[unique_ids] = fused_ownership
    # Match the single-view PDE's physical length control: use a metric band
    # around the hair mesh rather than a resolution-dependent voxel dilation.
    seg_support_volume = build_multiview_seg_support_volume(
        geometry_calibs,
        seg_masks,
        b_min,
        b_max,
        args.pde_resolution,
        head_mesh_path="data/head_model.obj",
        occluder_mesh_path=aligned_surface_path,
        visibility_tolerance=args.head_occlusion_tolerance,
    )
    boundary_mask &= seg_support_volume
    fused_orien[:, ~boundary_mask] = 0.0
    view_ownership[~boundary_mask] = -1
    hair_volume = build_mesh_metric_band(
        aligned_surface_path,
        seg_support_volume,
        b_min,
        b_max,
        band_width=args.pde_band_width,
    )
    hair_volume |= boundary_mask
    print(
        f"  Visible outer shell: raw={raw_surface_shell.sum()}, "
        f"metric_band={hair_volume.sum()}, per_view={shell_counts}; "
        f"fused contribution voxels={boundary_mask.sum()}"
    )

    # Save fusion and outer-shell debug data.
    np.savez_compressed(
        os.path.join(out_dir, "fusion_debug.npz"),
        fused_orien=fused_orien,
        boundary_mask=boundary_mask,
        view_ownership=view_ownership,
        raw_surface_shell=raw_surface_shell,
        seg_support_volume=seg_support_volume,
        hair_volume=hair_volume,
    )
    print(f"  Fusion debug saved.")

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
        os.path.join(strand_dir, "front.png")
    ).astype(np.float32) / 255.0 * 2.0 - 1.0

    hairstep = np.concatenate([
        front_strand_bgr.transpose(2, 0, 1),  # (3, 512, 512) BGR in [-1,1]
        front_depth[None, :, :],
    ], axis=0)

    data = {
        "hairstep": torch.from_numpy(hairstep).float(),
        "calib": torch.from_numpy(geometry_calibs["front"][0]).float().clone(),
    }

    b_min_val = np.asarray(b_min, dtype=np.float32)
    b_max_val = np.asarray(b_max, dtype=np.float32)

    # Keep the original strategy for a true front-only run. As soon as another
    # view is present, consume the fused boundary built in Step 3 instead of
    # silently rebuilding an orientation field from front data only.
    from lib.recon_strategy.laplace_pde import LaplacePDEStrategy

    opt_pde = argparse.Namespace()
    opt_pde.pde_resolution = args.pde_resolution
    opt_pde.pde_dilation_iters = args.pde_dilation_iters
    opt_pde.pde_cg_tol = args.pde_cg_tol
    opt_pde.pde_cg_maxiter = args.pde_cg_maxiter
    opt_pde.pde_anisotropy = args.pde_anisotropy
    opt_pde.pde_max_normal_component = args.pde_max_normal_component
    opt_pde.pde_harmonic_relax_iters = args.pde_harmonic_relax_iters
    opt_pde.pde_scalp_boundary_width = args.pde_scalp_boundary_width
    opt_pde.pde_alpha_mix = args.pde_alpha_mix
    opt_pde.pde_beta_mix = args.pde_beta_mix
    opt_pde.b_min = b_min_val
    opt_pde.b_max = b_max_val

    has_side_views = any(v != "front" for v in valid_views)
    if has_side_views:
        strategy = MultiViewLaplacePDEStrategy(opt_pde, cuda)
        strategy.set_fused_data(
            fused_orien,
            boundary_mask,
            hair_volume=hair_volume,
            view_ownership=view_ownership,
            head_mesh_path="data/head_model.obj",
        )
        print("  Using fused multi-view orientation boundary")
    else:
        strategy = LaplacePDEStrategy(opt_pde, cuda)
        print("  Using original front-only orientation field")
    strategy.filter(
        data,
        mesh_path=aligned_surface_path if has_side_views else args.mesh_obj,
    )
    strategy.set_query_mode("orien")

    # Keep the solved orientation inside the narrow PDE domain. Expanding it
    # with nearest-neighbor EDT would introduce discontinuities into the field.
    np.save(
        os.path.join(out_dir, "debug_orien_vol.npy"),
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
    from lib.hair_util import (
        get_hair_root,
        save_polyline_strands,
        save_strands_with_mesh,
    )
    root_tensor = (
        torch.from_numpy(get_hair_root(args.roots))
        .float()
        .unsqueeze(0)
        .to(cuda)
    )
    calib_tensor = (
        torch.from_numpy(geometry_calibs["front"][0]).float().unsqueeze(0).to(cuda)
    )
    root_calib_tensor = (
        torch.from_numpy(geometry_calibs["front"][0])
        .float()
        .unsqueeze(0)
        .to(cuda)
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
    # Root selection belongs to the head geometry, not to the hair-only image
    # correction.  Moving this projection would slide scalp roots with the
    # hairstyle and undo the separation between head pose and hair alignment.
    uv = torch.matmul(root_calib_tensor.squeeze(0), pts_homo)
    uv = uv[:2, :] / (uv[3:4, :] + 1e-8)
    uv_px_float = (uv + 1.0) * 0.5 * 511
    root_inside = (
        (uv_px_float[0] >= 0) & (uv_px_float[0] <= 511)
        & (uv_px_float[1] >= 0) & (uv_px_float[1] <= 511)
    )
    uv_px = uv_px_float.long().clamp(0, 511)
    front_visible_np = compute_root_head_visibility(
        "data/head_model.obj",
        roots_3d.T.detach().cpu().numpy(),
        geometry_calibs["front"][0],
    )
    front_visible = torch.from_numpy(front_visible_np).to(cuda)
    root_hair_mask = torch.zeros(N_roots, dtype=torch.bool, device=cuda)
    valid_pixels = torch.where(root_inside)[0]
    root_hair_mask[valid_pixels] = (
        torch.from_numpy(mask).to(cuda)[
            uv_px[1, valid_pixels], uv_px[0, valid_pixels]
        ] > 0.5
    )
    selectable_roots = root_inside & front_visible & root_hair_mask
    selectable_ids = torch.where(selectable_roots)[0]
    if len(selectable_ids) == 0:
        raise RuntimeError("No front-visible roots overlap the front hair mask")

    uv_np = uv_px_float[:, selectable_ids].T.detach().cpu().numpy()
    num_guides = min(1024, len(selectable_ids))
    kmeans = MiniBatchKMeans(
        n_clusters=num_guides, random_state=42, n_init="auto", batch_size=2048
    ).fit(uv_np)
    centroids_list = kmeans.cluster_centers_.tolist()

    guide_idx_list = []
    for cx, cy in centroids_list:
        candidate_uv = uv_px_float[:, selectable_ids]
        dist_sq = (candidate_uv[0] - cx) ** 2 + (candidate_uv[1] - cy) ** 2
        guide_idx_list.append(selectable_ids[torch.argmin(dist_sq)].item())

    guide_idx_tensor = torch.tensor(guide_idx_list, device=cuda)
    guide_roots = root_tensor[:, :, guide_idx_tensor]

    centroids_t = torch.tensor(centroids_list, device=cuda)

    # Nearest-centroid mapping for all roots
    dist_sq_all = (
        (uv_px_float[0].unsqueeze(0) - centroids_t[:, 0].unsqueeze(1)) ** 2
        + (uv_px_float[1].unsqueeze(0) - centroids_t[:, 1].unsqueeze(1)) ** 2
    )
    guide_indices_t = torch.argmin(dist_sq_all, dim=0)

    # Trace guide strands
    print("  Tracing 1024 guide strands...")
    guide_strands = hair_synthesis_rk4(
        strategy, cuda, guide_roots, calib_tensor,
        num_sample=args.num_sample, hair_unit=args.hair_unit,
        fallback_steps=args.pde_fallback_steps,
        sdf_vol=sdf_vol, normal_vol=normal_vol,
        b_min_t=b_min_t, b_max_t=b_max_t,
        silhouette_guard=silhouette_guard,
        silhouette_grace_steps=args.silhouette_grace_steps,
        max_turn_degrees=args.rk4_max_turn_degrees,
        actual_displacement_feedback=args.rk4_actual_displacement_feedback,
        label="guides",
    )
    guide_strands_t = (
        torch.from_numpy(guide_strands).to(cuda).permute(1, 2, 0)
    )

    # Trace all strands with guide clustering
    print(f"  Tracing {N_roots} full strands...")
    valid_cluster_mask = selectable_roots.float()

    strands = hair_synthesis_rk4(
        strategy, cuda, root_tensor, calib_tensor,
        num_sample=args.num_sample, hair_unit=args.hair_unit,
        fallback_steps=args.pde_fallback_steps,
        sdf_vol=sdf_vol, normal_vol=normal_vol,
        b_min_t=b_min_t, b_max_t=b_max_t,
        noise_vol=noise_vol,
        guide_strands=guide_strands_t,
        guide_indices=guide_indices_t,
        div_map_t=div_map_t,
        valid_cluster_mask=valid_cluster_mask,
        silhouette_guard=silhouette_guard,
        silhouette_grace_steps=args.silhouette_grace_steps,
        max_turn_degrees=args.rk4_max_turn_degrees,
        actual_displacement_feedback=args.rk4_actual_displacement_feedback,
        label="strands",
    )

    # ================================================================
    #  7. Save
    # ================================================================
    print(f"\nStep 7: Clipping & saving to {out_ply}...")
    if args.strand_smoothing_iters > 0:
        strands = smooth_strands_laplacian(
            strands,
            iterations=args.strand_smoothing_iters,
            strength=args.strand_smoothing_strength,
        )

        # Smoothing can move scalp-adjacent points into the head. Re-apply the
        # same continuous SDF correction used during RK4 before silhouette trim.
        flat = torch.from_numpy(strands.reshape(-1, 3).T).float().to(cuda)
        corrected = 0
        for start in range(0, flat.shape[1], 200_000):
            stop = min(start + 200_000, flat.shape[1])
            points = flat[:, start:stop]
            sdf = query_grid(sdf_vol, points, b_min_t, b_max_t)
            normal = torch.nn.functional.normalize(
                query_grid(normal_vol, points, b_min_t, b_max_t), dim=0
            )
            penetration = 0.005 - sdf
            collision = penetration > 0.0
            points = points + collision * penetration * normal
            flat[:, start:stop] = points
            corrected += int(collision.sum())
        strands = flat.T.reshape(strands.shape).cpu().numpy()
        print(
            f"[Strand Smoothing] iterations={args.strand_smoothing_iters}, "
            f"strength={args.strand_smoothing_strength:.2f}, "
            f"collision-corrected points={corrected}"
        )

    if silhouette_guard is not None:
        from lib.hair_util import trim_strands_by_silhouette

        strands = trim_strands_by_silhouette(
            strands,
            silhouette_guard,
            trim_margin_px=args.silhouette_trim_margin_px,
        )
    save_strands_with_mesh(strands, args.mesh_obj, out_ply, 0.3, is_eval=False)

    if args.export_per_view:
        print("\nStep 8: Exporting independent per-view strands...")
        per_view_root = os.path.join(out_dir, "per_view")
        fused_out_dir = os.path.join(per_view_root, "fused")
        os.makedirs(fused_out_dir, exist_ok=True)
        save_strands_with_mesh(
            strands,
            args.mesh_obj,
            os.path.join(fused_out_dir, "hair.ply"),
            0.3,
            is_eval=False,
        )
        for view in valid_views:
            curves = trace_view_strands_3d(
                strand_maps[view], depth_maps[view], geometry_calibs[view][0]
            )
            view_out_dir = os.path.join(per_view_root, view)
            os.makedirs(view_out_dir, exist_ok=True)
            depth_out_path = os.path.join(
                view_out_dir, "depth_backprojected_hair.ply"
            )
            save_polyline_strands(curves, depth_out_path)

            contribution_points, contribution_lines = contribution_records[view]
            contribution = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(contribution_points),
                lines=o3d.utility.Vector2iVector(contribution_lines),
            )
            color = {
                "front": [1.0, 0.12, 0.08],
                "left": [0.08, 0.30, 1.0],
                "right": [0.08, 0.75, 0.25],
                "back": [0.75, 0.15, 0.85],
            }.get(view, [0.8, 0.8, 0.8])
            contribution.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(contribution_lines), 1))
            )
            view_out_path = os.path.join(view_out_dir, "hair.ply")
            contribution_out_dir = os.path.join(out_dir, "view_contributions")
            os.makedirs(contribution_out_dir, exist_ok=True)
            contribution_out_path = os.path.join(
                contribution_out_dir, f"{view}.ply"
            )
            if not o3d.io.write_line_set(view_out_path, contribution):
                raise RuntimeError(f"Failed to save {view_out_path}")
            if not o3d.io.write_line_set(contribution_out_path, contribution):
                raise RuntimeError(f"Failed to save {contribution_out_path}")
            print(
                f"  {view}: {len(contribution_lines)} mesh-surface contribution "
                f"vectors -> {view_out_path}; {len(curves)} depth curves -> "
                f"{depth_out_path}"
            )

    print(f"Done → {out_ply}")


if __name__ == "__main__":
    main()
