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
import json

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
    build_surface_attraction_volume,
    build_visible_hair_proxy_mesh,
    build_visible_mesh_surface_shell,
    compute_root_head_visibility,
    extract_view_mesh_surface_contribution,
    fuse_multiview_orientation,
    orient_sparse_direction_axes,
    save_mesh_root_projections,
    trace_view_strands_3d,
)
from lib.multiview_pde import MultiViewLaplacePDEStrategy
from lib.recon_strategy.weighted_poisson import (
    build_signed_domain_distance,
    limit_outward_domain_direction,
    recover_points_inside_domain,
)


def query_grid(vol_5d, points_3n, b_min_tensor, b_max_tensor):
    """Sample a 3D volume at world-space points."""
    points_3n = points_3n.reshape(3, -1)
    b_min_tensor = b_min_tensor.reshape(3, 1)
    b_max_tensor = b_max_tensor.reshape(3, 1)
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


def parting_back_taper_weight(image_y, active_y_min, taper_px):
    """Smoothly open the parting from its rear/image-top endpoint."""
    if float(taper_px) <= 0.0:
        return torch.ones_like(image_y)
    t = torch.clamp(
        (image_y - float(active_y_min)) / float(taper_px), 0.0, 1.0
    )
    return t * t * (3.0 - 2.0 * t)


def root_collision_distance_for_step(
    step, ramp_steps=0, start_distance=0.0015, full_distance=0.005
):
    """根段碰撞壳从贴头皮距离平滑过渡到常规安全距离。"""
    steps = max(0, int(ramp_steps))
    if steps <= 0 or int(step) > steps:
        return float(full_distance)
    if steps == 1:
        return float(start_distance)
    t = np.clip((int(step) - 1) / float(steps - 1), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    return float(start_distance) + weight * (
        float(full_distance) - float(start_distance)
    )


def apply_parting_direction_barrier(
    direction,
    points,
    calib,
    center_x_by_row,
    root_side_sign,
    image_height,
    image_width,
    active_y_min,
    active_y_max,
    radius_px,
    min_outward_component=0.05,
    back_taper_px=0.0,
    back_cap_radius_px=0.0,
):
    """Continuously keep near-parting RK4 velocity on its root side.

    The PDE stores an unoriented line field. Sign propagation makes a strand
    locally continuous, but it does not make the detected parting a topological
    separator. This barrier corrects velocity, not completed curve points, and
    activates only while a sample approaches the curved centerline.
    """
    if float(radius_px) <= 0.0 or direction.numel() == 0:
        inactive = torch.zeros(
            direction.shape[1], dtype=torch.bool, device=direction.device
        )
        return direction, inactive, inactive
    matrix = calib[0] if calib.ndim == 3 else calib
    height = int(image_height)
    width = int(image_width)
    homogeneous = torch.cat(
        [points, torch.ones(1, points.shape[1], device=points.device)], dim=0
    )
    clip = matrix @ homogeneous
    ndc = clip[:2] / torch.clamp(clip[3:4], min=1e-8)
    px = (ndc[0] + 1.0) * 0.5 * max(width - 1, 1)
    py = (ndc[1] + 1.0) * 0.5 * max(height - 1, 1)
    rows = torch.round(py).long().clamp(0, height - 1)
    side = root_side_sign.reshape(-1).to(direction.dtype)
    boundary_rows = center_x_by_row
    taper = parting_back_taper_weight(py, active_y_min, back_taper_px)
    if boundary_rows.ndim == 1:
        boundary = boundary_rows[rows]
        boundary_before = boundary_rows[(rows - 1).clamp(0, height - 1)]
        boundary_after = boundary_rows[(rows + 1).clamp(0, height - 1)]
    else:
        side_index = (side > 0.0).long()
        raw_boundary = boundary_rows[side_index, rows]
        raw_boundary_before = boundary_rows[
            side_index, (rows - 1).clamp(0, height - 1)
        ]
        raw_boundary_after = boundary_rows[
            side_index, (rows + 1).clamp(0, height - 1)
        ]
        center = 0.5 * (boundary_rows[0, rows] + boundary_rows[1, rows])
        center_before = 0.5 * (
            boundary_rows[0, (rows - 1).clamp(0, height - 1)]
            + boundary_rows[1, (rows - 1).clamp(0, height - 1)]
        )
        center_after = 0.5 * (
            boundary_rows[0, (rows + 1).clamp(0, height - 1)]
            + boundary_rows[1, (rows + 1).clamp(0, height - 1)]
        )
        taper_before = parting_back_taper_weight(
            py - 1.0, active_y_min, back_taper_px
        )
        taper_after = parting_back_taper_weight(
            py + 1.0, active_y_min, back_taper_px
        )
        boundary = center + taper * (raw_boundary - center)
        boundary_before = center_before + taper_before * (
            raw_boundary_before - center_before
        )
        boundary_after = center_after + taper_after * (
            raw_boundary_after - center_after
        )
    center_slope = 0.5 * (boundary_after - boundary_before)
    signed_distance = side * (px - boundary)
    side_active = (
        (side != 0.0)
        & (taper > 1e-4)
        & (py >= float(active_y_min))
        & (py <= float(active_y_max))
        & (signed_distance <= float(radius_px))
    )
    cap_radius = max(0.0, float(back_cap_radius_px))
    cap_center_x = 0.5 * (
        boundary_rows[0, int(active_y_min)]
        + boundary_rows[1, int(active_y_min)]
    ) if boundary_rows.ndim > 1 else boundary_rows[int(active_y_min)]
    cap_dx = px - cap_center_x
    cap_dy = py - float(active_y_min)
    cap_distance = torch.sqrt(cap_dx * cap_dx + cap_dy * cap_dy)
    cap_linear = torch.clamp(
        1.0 - cap_distance / max(cap_radius, 1e-8), 0.0, 1.0
    )
    cap_weight = cap_linear * cap_linear * (3.0 - 2.0 * cap_linear)
    cap_active = (
        (side != 0.0)
        & (cap_radius > 0.0)
        & (py < float(active_y_min))
        & (cap_distance < cap_radius)
    )
    active = side_active | cap_active
    if not active.any():
        return direction, active, cap_active

    # Unproject a one-pixel step along the curved side normal or rounded rear
    # cap normal. The cap rotates the constraint toward image-top/posterior,
    # instead of letting the lateral component disappear and bridge the seam.
    cap_safe_distance = torch.clamp(cap_distance, min=1e-6)
    cap_normal_x = cap_dx / cap_safe_distance
    cap_normal_y = cap_dy / cap_safe_distance
    cap_center = cap_distance < 1e-6
    cap_normal_x = torch.where(cap_center, side, cap_normal_x)
    cap_normal_y = torch.where(
        cap_center, torch.full_like(cap_normal_y, -1.0), cap_normal_y
    )
    normal_x = torch.where(cap_active, cap_normal_x, side)
    normal_y = torch.where(
        cap_active, cap_normal_y, -side * center_slope
    )
    shifted_clip = clip.clone()
    shifted_clip[0] += normal_x * (2.0 / max(width - 1, 1)) * clip[3]
    shifted_clip[1] += normal_y * (2.0 / max(height - 1, 1)) * clip[3]
    inverse = torch.linalg.inv(matrix)
    shifted_homogeneous = inverse @ shifted_clip
    shifted_points = shifted_homogeneous[:3] / torch.clamp(
        shifted_homogeneous[3:4], min=1e-8
    )
    outward = torch.nn.functional.normalize(shifted_points - points, dim=0)
    magnitude = torch.norm(direction, dim=0, keepdim=True)
    component = torch.sum(direction * outward, dim=0, keepdim=True)
    strength = torch.where(cap_active, cap_weight, taper)
    target = float(min_outward_component) * strength.unsqueeze(0) * magnitude
    needs_correction = active & (component.squeeze(0) < target.squeeze(0))
    correction = torch.clamp(target - component, min=0.0) * outward
    corrected = torch.where(
        needs_correction.unsqueeze(0), direction + correction, direction
    )
    return corrected, needs_correction, cap_active


def project_parting_candidate_to_side(
    points,
    calib,
    boundary_x_by_row,
    root_side_sign,
    image_height,
    image_width,
    active_y_min,
    active_y_max,
    margin_px=1.0,
    back_taper_px=0.0,
):
    """Minimally project a crossed RK4 candidate back to its mask edge."""
    matrix = calib[0] if calib.ndim == 3 else calib
    height = int(image_height)
    width = int(image_width)
    side = root_side_sign.reshape(-1).to(points.dtype)
    homogeneous = torch.cat(
        [points, torch.ones(1, points.shape[1], device=points.device)], dim=0
    )
    clip = matrix @ homogeneous
    ndc = clip[:2] / torch.clamp(clip[3:4], min=1e-8)
    px = (ndc[0] + 1.0) * 0.5 * max(width - 1, 1)
    py = (ndc[1] + 1.0) * 0.5 * max(height - 1, 1)
    rows = torch.round(py).long().clamp(0, height - 1)
    side_index = (side > 0.0).long()
    raw_boundary = boundary_x_by_row[side_index, rows]
    center = 0.5 * (
        boundary_x_by_row[0, rows] + boundary_x_by_row[1, rows]
    )
    taper = parting_back_taper_weight(py, active_y_min, back_taper_px)
    boundary = center + taper * (raw_boundary - center)
    clearance = side * (px - boundary)
    effective_margin = taper * float(margin_px)
    violated = (
        (side != 0.0)
        & (taper > 1e-4)
        & (py >= float(active_y_min))
        & (py <= float(active_y_max))
        & (clearance < effective_margin)
    )
    if not violated.any():
        return points, violated
    target_px = boundary + side * effective_margin
    target_ndc_x = target_px / max(width - 1, 1) * 2.0 - 1.0
    corrected_clip = clip.clone()
    corrected_clip[0] = torch.where(
        violated, target_ndc_x * clip[3], corrected_clip[0]
    )
    inverse = torch.linalg.inv(matrix)
    corrected_homogeneous = inverse @ corrected_clip
    corrected_points = corrected_homogeneous[:3] / torch.clamp(
        corrected_homogeneous[3:4], min=1e-8
    )
    return torch.where(violated.unsqueeze(0), corrected_points, points), violated


def enforce_parting_interface_candidate(
    candidate,
    step_origin,
    root_side_sign,
    interface_vol,
    interface_normal_vol,
    b_min_t,
    b_max_t,
    alive=None,
    occupancy_threshold=0.05,
):
    """将命中三维发缝界面的 RK4 候选点推回其发根所在侧。"""
    interface_value = query_grid(
        interface_vol, candidate, b_min_t, b_max_t
    ).reshape(-1)
    side = root_side_sign.reshape(-1).to(candidate.dtype)
    hit = (side != 0.0) & (interface_value > float(occupancy_threshold))
    if alive is not None:
        hit &= alive
    if not hit.any():
        return candidate, hit, torch.zeros_like(hit)

    interface_normal = torch.nn.functional.normalize(
        query_grid(
            interface_normal_vol, candidate, b_min_t, b_max_t
        ),
        dim=0,
    )
    allowed_normal = interface_normal * side.reshape(1, -1)
    displacement = candidate - step_origin
    displacement_length = torch.norm(displacement, dim=0, keepdim=True)
    allowed_component = torch.sum(
        displacement * allowed_normal, dim=0, keepdim=True
    )
    correction = torch.clamp(
        displacement_length - allowed_component, min=0.0
    ) * allowed_normal
    corrected = step_origin + displacement + correction
    corrected = torch.where(hit.unsqueeze(0), corrected, candidate)
    still_inside = (
        query_grid(
            interface_vol, corrected, b_min_t, b_max_t
        ).reshape(-1)
        > float(occupancy_threshold)
    ) & hit
    corrected = torch.where(
        still_inside.unsqueeze(0), step_origin, corrected
    )
    return corrected, hit, still_inside


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
    root_tangent_steps=0,
    root_tangent_strength=0.8,
    root_surface_follow_mask=None,
    root_surface_follow_steps=0,
    root_surface_follow_distance=0.001,
    surface_attraction_vol=None,
    surface_attraction_weight=0.0,
    surface_attraction_deadzone=0.015,
    surface_attraction_full_distance=0.05,
    depth_bias_direction=None,
    depth_bias_weight=0.0,
    depth_layer_offset=None,
    domain_sdf_vol=None,
    domain_normal_vol=None,
    domain_guard_margin=0.0,
    domain_guard_inward_bias=0.05,
    domain_projection_epsilon=0.001,
    domain_projection_enabled=False,
    local_domain_recovery=False,
    local_domain_recovery_margin=0.0,
    local_domain_recovery_max_angle_degrees=45.0,
    root_direction_reference=None,
    root_direction_reference_steps=0,
    root_direction_min_component=0.15,
    parting_barrier_center=None,
    parting_barrier_side=None,
    parting_barrier_y_bounds=None,
    parting_barrier_image_shape=None,
    parting_barrier_radius_px=0.0,
    parting_barrier_min_component=0.05,
    parting_barrier_margin_px=1.0,
    parting_barrier_back_taper_px=0.0,
    parting_barrier_back_cap_radius_px=0.0,
    parting_interface_vol=None,
    parting_interface_normal_vol=None,
    root_collision_ramp_steps=0,
    root_collision_start_distance=0.0015,
    return_diagnostics=False,
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
    first_low_field_step = torch.full(
        (num_strands,), -1, dtype=torch.long, device=cuda
    )
    first_domain_exit_step = torch.full_like(first_low_field_step, -1)
    first_silhouette_step = torch.full_like(first_low_field_step, -1)
    silhouette_hard_total = 0
    silhouette_grace_total = 0
    domain_projection_total = 0
    parting_barrier_total = 0
    parting_barrier_projection_total = 0
    parting_interface_total = 0
    parting_interface_freeze_total = 0
    depth_bias_weight_t = None
    if depth_bias_direction is not None:
        if torch.is_tensor(depth_bias_weight):
            depth_bias_weight_t = depth_bias_weight.to(
                device=cuda, dtype=torch.float32
            ).reshape(1, -1)
            if depth_bias_weight_t.shape[1] != num_strands:
                raise ValueError(
                    "Per-strand depth bias must match the number of roots"
                )
        elif abs(float(depth_bias_weight)) > 0.0:
            depth_bias_weight_t = torch.full(
                (1, num_strands),
                float(depth_bias_weight),
                dtype=torch.float32,
                device=cuda,
            )
    depth_layer_offset_t = None
    if depth_bias_direction is not None and depth_layer_offset is not None:
        if torch.is_tensor(depth_layer_offset):
            depth_layer_offset_t = depth_layer_offset.to(
                device=cuda, dtype=torch.float32
            ).reshape(1, -1)
        else:
            depth_layer_offset_t = torch.full(
                (1, num_strands),
                float(depth_layer_offset),
                dtype=torch.float32,
                device=cuda,
            )
        if depth_layer_offset_t.shape[1] != num_strands:
            raise ValueError(
                "Per-strand depth offset must match the number of roots"
            )
    offset_ramps = [
        min(1.0, max(0.0, (step / float(num_sample) - 0.05) / 0.30))
        for step in range(1, num_sample)
    ]
    offset_ramp_sum = max(sum(offset_ramps), 1e-8)

    def normalize_nonzero(vectors):
        norms = torch.norm(vectors, dim=0, keepdim=True)
        return torch.where(
            norms > 1e-8,
            vectors / torch.clamp(norms, min=1e-8),
            vectors,
        )

    def query_direction(
        points,
        fallback_mask,
        reference_direction=None,
        return_low=False,
    ):
        direction = strategy.query(points, calib_tensor).squeeze(0)
        low = torch.norm(direction, dim=0) < 0.05
        if (
            local_domain_recovery
            and domain_sdf_vol is not None
            and domain_normal_vol is not None
            and low.any()
        ):
            sample_points = points.squeeze(0)
            domain_sdf = query_grid(
                domain_sdf_vol, sample_points, b_min_t, b_max_t
            ).squeeze(0)
            domain_normal = query_grid(
                domain_normal_vol, sample_points, b_min_t, b_max_t
            )
            recovered_points = recover_points_inside_domain(
                sample_points,
                domain_sdf,
                domain_normal,
                local_domain_recovery_margin,
                domain_projection_epsilon,
            ).unsqueeze(0)
            recovered = strategy.query(
                recovered_points, calib_tensor
            ).squeeze(0)
            recovered_valid = torch.norm(recovered, dim=0) >= 0.05
            recoverable = domain_sdf >= -float(local_domain_recovery_margin)
            if reference_direction is not None:
                reference_unit = normalize_nonzero(reference_direction)
                recovered_unit = normalize_nonzero(recovered)
                alignment = torch.abs(
                    torch.sum(reference_unit * recovered_unit, dim=0)
                )
                angle_limit = torch.cos(
                    torch.deg2rad(
                        torch.tensor(
                            local_domain_recovery_max_angle_degrees,
                            device=cuda,
                        )
                    )
                )
                recovered_valid = recovered_valid & (alignment >= angle_limit)
            use_recovered = low & recoverable & recovered_valid
            direction = torch.where(
                use_recovered.unsqueeze(0), recovered, direction
            )
            low = torch.norm(direction, dim=0) < 0.05
        use_fallback = low & fallback_mask
        if use_fallback.any() and hasattr(strategy, "query_fallback"):
            fallback = strategy.query_fallback(points, calib_tensor).squeeze(0)
            direction = torch.where(use_fallback.unsqueeze(0), fallback, direction)
        if domain_sdf_vol is not None and domain_normal_vol is not None:
            sample_points = points.squeeze(0)
            domain_sdf = query_grid(
                domain_sdf_vol, sample_points, b_min_t, b_max_t
            ).squeeze(0)
            domain_normal = query_grid(
                domain_normal_vol, sample_points, b_min_t, b_max_t
            )
            direction = limit_outward_domain_direction(
                direction,
                domain_sdf,
                domain_normal,
                domain_guard_margin,
                domain_guard_inward_bias,
            )
        return (direction, low) if return_low else direction

    for index in range(1, num_sample):
        step_origin = current
        fallback_mask = (index <= int(fallback_steps)) | need_fallback
        k1, low_field = query_direction(
            current.unsqueeze(0),
            fallback_mask,
            reference_direction=previous_direction,
            return_low=True,
        )
        newly_low = alive & low_field & (first_low_field_step < 0)
        first_low_field_step[newly_low] = index
        if index == 1 and root_direction_reference is not None:
            k1 = align_vector_sign(k1, root_direction_reference)
        if previous_direction is not None:
            k1 = align_vector_sign(k1, previous_direction)
        k2 = query_direction(
            (current + 0.5 * hair_unit * k1).unsqueeze(0),
            fallback_mask,
            reference_direction=k1,
        )
        k2 = align_vector_sign(k2, k1)
        k3 = query_direction(
            (current + 0.5 * hair_unit * k2).unsqueeze(0),
            fallback_mask,
            reference_direction=k2,
        )
        k3 = align_vector_sign(k3, k2)
        k4 = query_direction(
            (current + hair_unit * k3).unsqueeze(0),
            fallback_mask,
            reference_direction=k3,
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
        if surface_attraction_vol is not None and surface_attraction_weight > 0.0:
            surface_pull = query_grid(
                surface_attraction_vol, current, b_min_t, b_max_t
            ).float()
            pull_distance = torch.norm(surface_pull, dim=0, keepdim=True)
            pull_direction = surface_pull / torch.clamp(
                pull_distance, min=1e-8
            )
            distance_span = max(
                float(surface_attraction_full_distance)
                - float(surface_attraction_deadzone),
                1e-6,
            )
            distance_weight = torch.clamp(
                (pull_distance - float(surface_attraction_deadzone))
                / distance_span,
                0.0,
                1.0,
            )
            # Keep roots governed by the scalp/PDE boundary, then introduce
            # the geometric correction gradually over the first third.
            attraction_ramp = min(1.0, max(0.0, (progress - 0.05) / 0.30))
            direction = direction + (
                float(surface_attraction_weight)
                * attraction_ramp
                * distance_weight
                * pull_direction
                * magnitude
            )
        if depth_bias_direction is not None and depth_bias_weight_t is not None:
            # Orthographic front projection is invariant along this ray. A
            # small ramped component therefore adds controlled 3D thickness
            # without changing the front-view strand direction.
            depth_ramp = min(1.0, max(0.0, (progress - 0.05) / 0.30))
            direction = direction + (
                depth_bias_weight_t
                * depth_ramp
                * depth_bias_direction
                * magnitude
            )
        if (
            root_tangent_steps > 0
            and index <= int(root_tangent_steps)
            and sdf_vol is not None
            and normal_vol is not None
        ):
            scalp_normal = torch.nn.functional.normalize(
                query_grid(normal_vol, current, b_min_t, b_max_t), dim=0
            )
            normal_component = torch.sum(direction * scalp_normal, dim=0, keepdim=True)
            tangent = direction - normal_component * scalp_normal
            magnitude = torch.norm(direction, dim=0, keepdim=True)
            outward = torch.clamp(normal_component, min=0.0)
            normal_bias = 0.15 * magnitude + 0.35 * outward
            ramp = min(1.0, max(0.0, float(index) / max(1.0, float(root_tangent_steps))))
            tangent_strength = float(root_tangent_strength) * (1.0 - 0.7 * ramp)
            direction = tangent_strength * tangent + normal_bias * scalp_normal
        if (
            root_direction_reference is not None
            and index <= int(root_direction_reference_steps)
        ):
            reference_unit = normalize_nonzero(root_direction_reference)
            direction_magnitude = torch.norm(direction, dim=0, keepdim=True)
            reference_component = torch.sum(
                direction * reference_unit, dim=0, keepdim=True
            )
            target_component = (
                float(root_direction_min_component) * direction_magnitude
            )
            correction = torch.clamp(
                target_component - reference_component, min=0.0
            )
            direction = direction + correction * reference_unit
        if (
            parting_barrier_center is not None
            and parting_barrier_side is not None
            and parting_barrier_y_bounds is not None
            and parting_barrier_image_shape is not None
            and float(parting_barrier_radius_px) > 0.0
        ):
            direction, barrier_corrected, parting_cap_active = (
                apply_parting_direction_barrier(
                    direction,
                    current,
                    calib_tensor,
                    parting_barrier_center,
                    parting_barrier_side,
                    parting_barrier_image_shape[0],
                    parting_barrier_image_shape[1],
                    parting_barrier_y_bounds[0],
                    parting_barrier_y_bounds[1],
                    parting_barrier_radius_px,
                    parting_barrier_min_component,
                    parting_barrier_back_taper_px,
                    parting_barrier_back_cap_radius_px,
                )
            )
            parting_barrier_total += int((barrier_corrected & alive).sum())
        else:
            parting_cap_active = torch.zeros_like(alive)
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
        if depth_layer_offset_t is not None:
            current = current + (
                depth_bias_direction
                * depth_layer_offset_t
                * (offset_ramps[index - 1] / offset_ramp_sum)
            )

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
            follow_surface = None
            if (
                root_surface_follow_mask is not None
                and index <= int(root_surface_follow_steps)
            ):
                follow_surface = root_surface_follow_mask.bool().reshape(-1)
            if parting_cap_active.any():
                follow_surface = (
                    parting_cap_active
                    if follow_surface is None
                    else (follow_surface | parting_cap_active)
                )
            if follow_surface is not None:
                correction = float(root_surface_follow_distance) - sdf
                current = current + (
                    follow_surface.float().unsqueeze(0)
                    * correction
                    * normal
                )
                sdf = torch.where(
                    follow_surface,
                    torch.full_like(sdf, float(root_surface_follow_distance)),
                    sdf,
                )
            collision_distance = torch.full_like(
                sdf,
                root_collision_distance_for_step(
                    index,
                    ramp_steps=root_collision_ramp_steps,
                    start_distance=root_collision_start_distance,
                    full_distance=0.005,
                ),
            )
            if follow_surface is not None:
                collision_distance = torch.where(
                    follow_surface,
                    torch.full_like(sdf, float(root_surface_follow_distance)),
                    collision_distance,
                )
            penetration = collision_distance - sdf
            collision_mask = (penetration > 0).float()
            current = current + collision_mask * penetration * normal

        if (
            parting_interface_vol is not None
            and parting_interface_normal_vol is not None
            and parting_barrier_side is not None
        ):
            current, interface_hit, interface_frozen = (
                enforce_parting_interface_candidate(
                    current,
                    step_origin,
                    parting_barrier_side,
                    parting_interface_vol,
                    parting_interface_normal_vol,
                    b_min_t,
                    b_max_t,
                    alive=alive,
                )
            )
            parting_interface_total += int(interface_hit.sum())
            parting_interface_freeze_total += int(interface_frozen.sum())

        if (
            parting_barrier_center is not None
            and parting_barrier_side is not None
            and parting_barrier_y_bounds is not None
            and parting_barrier_image_shape is not None
            and float(parting_barrier_radius_px) > 0.0
        ):
            current, projected_to_side = project_parting_candidate_to_side(
                current,
                calib_tensor,
                parting_barrier_center,
                parting_barrier_side,
                parting_barrier_image_shape[0],
                parting_barrier_image_shape[1],
                parting_barrier_y_bounds[0],
                parting_barrier_y_bounds[1],
                margin_px=parting_barrier_margin_px,
                back_taper_px=parting_barrier_back_taper_px,
            )
            parting_barrier_projection_total += int(
                (projected_to_side & alive).sum()
            )

        candidate_domain_sdf = None
        candidate_domain_normal = None
        if domain_sdf_vol is not None and domain_normal_vol is not None:
            candidate_domain_sdf = query_grid(
                domain_sdf_vol, current, b_min_t, b_max_t
            ).squeeze(0)
            candidate_domain_normal = query_grid(
                domain_normal_vol, current, b_min_t, b_max_t
            )
            outside_domain = candidate_domain_sdf < 0.0
        else:
            occupancy = strategy.query_occ(
                current.unsqueeze(0), calib_tensor
            ).squeeze(0).squeeze(0)
            outside_domain = occupancy < 0.5
        if return_diagnostics:
            newly_outside = (
                alive & outside_domain & (first_domain_exit_step < 0)
            )
            first_domain_exit_step[newly_outside] = index
        if candidate_domain_sdf is not None and domain_projection_enabled:
            project_domain = alive & outside_domain
            if project_domain.any():
                inward_normal = torch.nn.functional.normalize(
                    candidate_domain_normal, dim=0
                )
                correction = torch.clamp(
                    -candidate_domain_sdf + float(domain_projection_epsilon),
                    min=0.0,
                )
                current = current + (
                    project_domain.float() * correction
                ).unsqueeze(0) * inward_normal
                domain_projection_total += int(project_domain.sum())

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
                newly_hard = hard
                newly_grace = newly_dead & ~hard
                alive = alive & ~newly_dead
                need_fallback = alive & soft
                dead_step = int(newly_dead.sum())
                dead_idx = torch.nonzero(
                    newly_dead, as_tuple=False
                ).squeeze(1)
                current = torch.where(alive.unsqueeze(0), current, hair_strands[index - 1])
            else:
                sub_grace = grace_left[check_idx]
                sub_grace = torch.where(
                    soft, sub_grace - 1, torch.full_like(sub_grace, grace_budget)
                )
                grace_left[check_idx] = sub_grace
                newly_dead_sub = hard | (soft & (sub_grace < 0))
                newly_hard = newly_dead_sub & hard
                newly_grace = newly_dead_sub & ~hard
                dead_idx = check_idx[newly_dead_sub]
                alive[dead_idx] = False
                need_fallback = torch.zeros_like(need_fallback)
                need_fallback[check_idx[soft & ~newly_dead_sub]] = True
                dead_step = int(newly_dead_sub.sum())
                if dead_step:
                    current[:, dead_idx] = hair_strands[index - 1, :, dead_idx]
            if dead_step:
                first_silhouette_step[dead_idx] = index
            silhouette_hard_total += int(newly_hard.sum())
            silhouette_grace_total += int(newly_grace.sum())
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

        # Enforce the parting invariant after every operation that may replace
        # the candidate (domain handling, silhouette rollback, or freezing).
        # The earlier projection shapes the live trajectory; this final pass
        # guarantees that the point actually written to the strand is valid.
        if (
            parting_barrier_center is not None
            and parting_barrier_side is not None
            and parting_barrier_y_bounds is not None
            and parting_barrier_image_shape is not None
            and float(parting_barrier_radius_px) > 0.0
        ):
            current, final_side_projection = project_parting_candidate_to_side(
                current,
                calib_tensor,
                parting_barrier_center,
                parting_barrier_side,
                parting_barrier_image_shape[0],
                parting_barrier_image_shape[1],
                parting_barrier_y_bounds[0],
                parting_barrier_y_bounds[1],
                margin_px=parting_barrier_margin_px,
                back_taper_px=parting_barrier_back_taper_px,
            )
            parting_barrier_projection_total += int(
                final_side_projection.sum()
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

    strands_np = hair_strands.permute(2, 0, 1).cpu().detach().numpy()
    if not return_diagnostics:
        return strands_np

    low_steps = first_low_field_step.cpu().numpy()
    domain_steps = first_domain_exit_step.cpu().numpy()
    silhouette_steps = first_silhouette_step.cpu().numpy()
    segment_lengths = np.linalg.norm(
        strands_np[:, 1:] - strands_np[:, :-1], axis=2
    )
    effective_points = 1 + (segment_lengths > 1e-7).sum(axis=1)

    def summarize_steps(steps):
        observed = steps[steps >= 0]
        if not len(observed):
            return {"count": 0, "q10": None, "q50": None, "q90": None}
        quantiles = np.quantile(observed, [0.1, 0.5, 0.9])
        return {
            "count": int(len(observed)),
            "q10": float(quantiles[0]),
            "q50": float(quantiles[1]),
            "q90": float(quantiles[2]),
        }

    silhouette_seen = silhouette_steps >= 0
    domain_before_silhouette = (domain_steps >= 0) & (
        ~silhouette_seen | (domain_steps <= silhouette_steps)
    )
    low_before_silhouette = (low_steps >= 0) & (
        ~silhouette_seen | (low_steps <= silhouette_steps)
    )
    effective_quantiles = np.quantile(effective_points, [0.1, 0.5, 0.9])
    diagnostics = {
        "label": label or "rk4",
        "num_strands": int(num_strands),
        "num_samples": int(num_sample),
        "first_low_field_step": summarize_steps(low_steps),
        "first_domain_exit_step": summarize_steps(domain_steps),
        "first_silhouette_step": summarize_steps(silhouette_steps),
        "silhouette_hard_stops": int(silhouette_hard_total),
        "silhouette_grace_stops": int(silhouette_grace_total),
        "domain_exit_before_silhouette": int(domain_before_silhouette.sum()),
        "low_field_before_silhouette": int(low_before_silhouette.sum()),
        "domain_projection_events": int(domain_projection_total),
        "parting_barrier_events": int(parting_barrier_total),
        "parting_barrier_projection_events": int(
            parting_barrier_projection_total
        ),
        "parting_interface_events": int(parting_interface_total),
        "parting_interface_freeze_events": int(
            parting_interface_freeze_total
        ),
        "non_silhouette_stalls": int(
            ((effective_points < num_sample) & ~silhouette_seen).sum()
        ),
        "effective_points_q10": float(effective_quantiles[0]),
        "effective_points_q50": float(effective_quantiles[1]),
        "effective_points_q90": float(effective_quantiles[2]),
    }
    print(f"  [{label or 'rk4'}] diagnostics: {diagnostics}")
    return strands_np, diagnostics


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


def build_front_parting_root_reference(
    roots,
    calib,
    strand_map,
    parting_mask,
    radius_px=96.0,
    scalp_normals=None,
    row_margin_before=24,
    row_margin_after=48,
):
    """从 front 局部发流构造远离发缝且贴头皮切向的 3D 根部参考。"""
    roots = np.asarray(roots, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    strand_map = np.asarray(strand_map, dtype=np.float32)
    mask = np.asarray(parting_mask, dtype=bool)
    height, width = mask.shape
    part_y, part_x = np.where(mask)
    reference = np.zeros_like(roots, dtype=np.float64)
    if len(part_x) == 0 or len(roots) == 0:
        return reference.astype(np.float32), np.zeros(len(roots), dtype=bool)

    homogeneous = np.column_stack([roots, np.ones(len(roots))])
    clip = homogeneous @ matrix.T
    uv = clip[:, :2] / (clip[:, 3:4] + 1e-12)
    depth = clip[:, 2] / (clip[:, 3] + 1e-12)
    px = (uv[:, 0] + 1.0) * 0.5 * (width - 1)
    py = (uv[:, 1] + 1.0) * 0.5 * (height - 1)
    inside = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)

    center_by_y = np.full(height, np.nan, dtype=np.float64)
    for row in np.unique(part_y):
        center_by_y[row] = float(np.median(part_x[part_y == row]))
    valid_rows = np.flatnonzero(np.isfinite(center_by_y))
    center_by_y = np.interp(
        np.arange(height), valid_rows, center_by_y[valid_rows]
    )
    rows = np.rint(py).astype(np.int64).clip(0, height - 1)
    signed_distance = px - center_by_y[rows]

    sample_x = px.astype(np.float32).reshape(-1, 1)
    sample_y = py.astype(np.float32).reshape(-1, 1)
    channel_b = cv2.remap(
        strand_map[:, :, 2], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1)
    channel_g = cv2.remap(
        strand_map[:, :, 1], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1)
    confidence = cv2.remap(
        strand_map[:, :, 0], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1)
    direction_x = 1.0 - 2.0 * channel_b
    direction_y = 2.0 * channel_g - 1.0

    inverse = np.linalg.inv(matrix)
    epsilon = 2.0 / max(height, width)

    def unproject(coords):
        local_clip = np.column_stack([
            coords,
            depth,
            np.ones(len(coords), dtype=np.float64),
        ])
        world = local_clip @ inverse.T
        return world[:, :3] / (world[:, 3:4] + 1e-12)

    base = unproject(uv)
    tangent_u = unproject(uv + np.array([epsilon, 0.0])) - base
    tangent_v = unproject(uv + np.array([0.0, epsilon])) - base
    reference = (
        direction_x[:, None] * tangent_u
        + direction_y[:, None] * tangent_v
    )
    if scalp_normals is not None:
        normals = np.asarray(scalp_normals, dtype=np.float64).reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        reference -= np.sum(reference * normals, axis=1, keepdims=True) * normals

    norm = np.linalg.norm(reference, axis=1)
    influence = (
        inside
        & (confidence > 0.1)
        & (np.abs(signed_distance) <= float(radius_px))
        & (py >= float(part_y.min() - int(row_margin_before)))
        & (py <= float(part_y.max() + int(row_margin_after)))
        & (norm > 1e-10)
        & (np.abs(signed_distance) >= 1.0)
    )
    reference[influence] /= norm[influence, None]
    screen_x_axis = matrix[0, :3]
    screen_x_axis /= np.linalg.norm(screen_x_axis) + 1e-12
    projected_side = reference @ screen_x_axis
    reverse = influence & (projected_side * np.sign(signed_distance) < 0.0)
    reference[reverse] *= -1.0
    reference[~influence] = 0.0
    return reference.astype(np.float32), influence


def build_scalp_parting_geodesic_reference(
    roots,
    calib,
    parting_mask,
    scalp_normals,
    head_mesh=None,
    radius_px=48.0,
    endpoint_blend_px=18.0,
    curve_neighbors=8,
    curve_mode="root_fit",
):
    """把 front 发缝提升到头皮，并构造左右分离的切向根方向。

    发缝曲线可由每一有效图像行附近的头皮根样本拟合，或从相机射线
    直接命中可见头皮得到。后者避免同一像素下前后头皮根混合。
    普通曲线段使用从发缝向两侧的近似测地方向；后冠端逐渐混合为
    沿三维头皮曲线向后的共同方向，从而避免二维横向端帽。
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    roots = np.asarray(roots, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(scalp_normals, dtype=np.float64).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    mask = np.asarray(parting_mask, dtype=bool)
    height, width = mask.shape
    reference = np.zeros_like(roots)
    influence = np.zeros(len(roots), dtype=bool)
    ys, xs = np.where(mask)
    if len(roots) == 0 or len(xs) == 0:
        return reference.astype(np.float32), influence, np.empty((0, 3))

    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    homogeneous = np.column_stack([roots, np.ones(len(roots))])
    clip = homogeneous @ matrix.T
    uv = clip[:, :2] / (clip[:, 3:4] + 1e-12)
    px = (uv[:, 0] + 1.0) * 0.5 * (width - 1)
    py = (uv[:, 1] + 1.0) * 0.5 * (height - 1)
    inside = (
        (px >= 0.0) & (px <= width - 1)
        & (py >= 0.0) & (py <= height - 1)
    )

    valid_rows = np.unique(ys)
    center_by_row = np.full(height, np.nan, dtype=np.float64)
    for row in valid_rows:
        center_by_row[row] = float(np.median(xs[ys == row]))
    center_by_row = np.interp(
        np.arange(height), valid_rows, center_by_row[valid_rows]
    )

    mesh = None
    if head_mesh is not None:
        if isinstance(head_mesh, o3d.geometry.TriangleMesh):
            mesh = trimesh.Trimesh(
                vertices=np.asarray(head_mesh.vertices),
                faces=np.asarray(head_mesh.triangles),
                process=False,
            )
        elif isinstance(head_mesh, trimesh.Trimesh):
            mesh = head_mesh
        else:
            mesh = trimesh.load(str(head_mesh), process=False)

    mode = str(curve_mode)
    curve = None
    if mode in {"mesh_raycast", "hybrid_raycast_cap"}:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("mesh_raycast 发缝曲线需要有效的头皮网格")
        inverse = np.linalg.inv(matrix)
        row_uv = np.column_stack([
            center_by_row[valid_rows] / max(width - 1, 1) * 2.0 - 1.0,
            valid_rows / max(height - 1, 1) * 2.0 - 1.0,
        ])

        def unproject_curve_depth(depth_value):
            local_clip = np.column_stack([
                row_uv,
                np.full(len(row_uv), depth_value, dtype=np.float64),
                np.ones(len(row_uv), dtype=np.float64),
            ])
            world = local_clip @ inverse.T
            return world[:, :3] / (world[:, 3:4] + 1e-12)

        ray_origins = unproject_curve_depth(-1.0)
        ray_ends = unproject_curve_depth(1.0)
        ray_directions = ray_ends - ray_origins
        ray_directions /= (
            np.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-12
        )
        locations, ray_ids, _ = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=True
        )
        curve = np.full((len(valid_rows), 3), np.nan, dtype=np.float64)
        for ray_id in range(len(valid_rows)):
            hits = locations[ray_ids == ray_id]
            if len(hits) == 0:
                continue
            distance = np.sum(
                (hits - ray_origins[ray_id]) * ray_directions[ray_id], axis=1
            )
            positive = distance >= 0.0
            if np.any(positive):
                positive_ids = np.flatnonzero(positive)
                chosen = positive_ids[np.argmin(distance[positive])]
            else:
                chosen = int(np.argmax(distance))
            curve[ray_id] = hits[chosen]
        valid_hits = np.all(np.isfinite(curve), axis=1)
        if np.sum(valid_hits) < 2:
            raise RuntimeError(
                "front 发缝射线与头皮网格交点不足，无法构造三维曲线"
            )
        curve_rows = np.arange(len(curve))
        for axis in range(3):
            curve[:, axis] = np.interp(
                curve_rows,
                curve_rows[valid_hits],
                curve[valid_hits, axis],
            )
    elif mode == "root_fit":
        sample_ids = np.flatnonzero(inside)
        sample_uv = np.column_stack([px[sample_ids], py[sample_ids]])
        neighbor_count = min(max(1, int(curve_neighbors)), len(sample_ids))
        curve = []
        for row in valid_rows:
            target = np.array([center_by_row[row], float(row)])
            distance_sq = np.sum((sample_uv - target) ** 2, axis=1)
            nearest = np.argpartition(
                distance_sq, neighbor_count - 1
            )[:neighbor_count]
            weight = 1.0 / np.maximum(distance_sq[nearest], 0.25)
            weight /= np.sum(weight)
            curve.append(
                np.sum(roots[sample_ids[nearest]] * weight[:, None], axis=0)
            )
        curve = np.asarray(curve, dtype=np.float64)
    else:
        raise ValueError(f"未知发缝三维曲线模式: {mode}")
    if len(curve) >= 5:
        curve = gaussian_filter1d(curve, sigma=1.5, axis=0, mode="nearest")

    if mesh is not None and len(curve):
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            curve, _, _ = trimesh.proximity.closest_point(mesh, curve)

    posterior_curve = curve
    lateral_curve = curve
    if mode == "hybrid_raycast_cap":
        sample_ids = np.flatnonzero(inside)
        sample_uv = np.column_stack([px[sample_ids], py[sample_ids]])
        neighbor_count = min(max(1, int(curve_neighbors)), len(sample_ids))
        root_curve = []
        for row in valid_rows:
            target = np.array([center_by_row[row], float(row)])
            distance_sq = np.sum((sample_uv - target) ** 2, axis=1)
            nearest = np.argpartition(
                distance_sq, neighbor_count - 1
            )[:neighbor_count]
            weight = 1.0 / np.maximum(distance_sq[nearest], 0.25)
            weight /= np.sum(weight)
            root_curve.append(
                np.sum(roots[sample_ids[nearest]] * weight[:, None], axis=0)
            )
        lateral_curve = np.asarray(root_curve, dtype=np.float64)
        if len(lateral_curve) >= 5:
            lateral_curve = gaussian_filter1d(
                lateral_curve, sigma=1.5, axis=0, mode="nearest"
            )
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            lateral_curve, _, _ = trimesh.proximity.closest_point(
                mesh, lateral_curve
            )

    nearest_curve = cKDTree(lateral_curve).query(roots, k=1)[1]
    lateral = roots - lateral_curve[nearest_curve]
    lateral -= np.sum(lateral * normals, axis=1, keepdims=True) * normals
    lateral_norm = np.linalg.norm(lateral, axis=1)

    posterior_span = min(4, len(posterior_curve) - 1)
    posterior = posterior_curve[0] - posterior_curve[posterior_span]
    posterior = np.broadcast_to(posterior, roots.shape).copy()
    posterior -= np.sum(posterior * normals, axis=1, keepdims=True) * normals
    posterior_norm = np.linalg.norm(posterior, axis=1)
    lateral_unit = lateral / np.maximum(lateral_norm[:, None], 1e-12)
    posterior_unit = posterior / np.maximum(posterior_norm[:, None], 1e-12)

    rows = np.rint(py).astype(np.int64).clip(0, height - 1)
    signed_distance = px - center_by_row[rows]
    endpoint = max(0.0, float(endpoint_blend_px))
    if endpoint > 0.0:
        blend = np.clip(
            1.0 - (py - float(valid_rows[0])) / endpoint, 0.0, 1.0
        )
        blend = blend * blend * (3.0 - 2.0 * blend)
    else:
        blend = np.zeros(len(roots), dtype=np.float64)
    reference = (
        (1.0 - blend[:, None]) * lateral_unit
        + blend[:, None] * posterior_unit
    )
    reference -= np.sum(reference * normals, axis=1, keepdims=True) * normals
    reference_norm = np.linalg.norm(reference, axis=1)
    influence = (
        inside
        & (np.abs(signed_distance) <= float(radius_px))
        & (np.abs(signed_distance) >= 1.0)
        & (py >= float(valid_rows[0]) - endpoint)
        & (py <= float(valid_rows[-1]) + float(radius_px))
        & (lateral_norm > 1e-8)
        & (posterior_norm > 1e-8)
        & (reference_norm > 1e-8)
    )
    reference[influence] /= reference_norm[influence, None]
    reference[~influence] = 0.0
    return reference.astype(np.float32), influence, curve.astype(np.float32)


def build_front_parting_barrier_data(roots, calib, parting_mask):
    """Build curved left/right mask edges and a persistent side per root."""
    roots = np.asarray(roots, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    mask = np.asarray(parting_mask, dtype=bool)
    height, width = mask.shape
    mask_y, mask_x = np.where(mask)
    left = np.full(height, np.nan, dtype=np.float64)
    right = np.full(height, np.nan, dtype=np.float64)
    side = np.zeros(len(roots), dtype=np.float32)
    if len(mask_x) == 0 or len(roots) == 0:
        return np.stack([left, right]).astype(np.float32), side, (0, -1)
    for row in np.unique(mask_y):
        row_x = mask_x[mask_y == row]
        left[row] = float(np.min(row_x))
        right[row] = float(np.max(row_x))
    valid_rows = np.flatnonzero(np.isfinite(left))
    left = np.interp(np.arange(height), valid_rows, left[valid_rows])
    right = np.interp(np.arange(height), valid_rows, right[valid_rows])
    center = 0.5 * (left + right)

    homogeneous = np.column_stack([roots, np.ones(len(roots))])
    clip = homogeneous @ matrix.T
    uv = clip[:, :2] / (clip[:, 3:4] + 1e-12)
    px = (uv[:, 0] + 1.0) * 0.5 * (width - 1)
    py = (uv[:, 1] + 1.0) * 0.5 * (height - 1)
    rows = np.rint(py).astype(np.int64).clip(0, height - 1)
    signed = px - center[rows]
    usable = (
        (px >= 0.0) & (px <= width - 1)
        & (py >= 0.0) & (py <= height - 1)
        & (np.abs(signed) >= 1.0)
    )
    side[usable] = np.sign(signed[usable]).astype(np.float32)
    return (
        np.stack([left, right]).astype(np.float32),
        side,
        (int(valid_rows[0]), int(valid_rows[-1])),
    )


def select_front_parting_back_cap_roots(
    roots, calib, boundary_x_by_row, active_y_min, radius_px, image_shape
):
    """选择发缝后冠圆角端帽内需要贴头皮前导的发根。"""
    roots = np.asarray(roots, dtype=np.float64).reshape(-1, 3)
    selected = np.zeros(len(roots), dtype=bool)
    radius = max(0.0, float(radius_px))
    if len(roots) == 0 or radius <= 0.0:
        return selected
    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    boundaries = np.asarray(boundary_x_by_row, dtype=np.float64)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    row = int(np.clip(active_y_min, 0, height - 1))
    center_x = 0.5 * (boundaries[0, row] + boundaries[1, row])
    homogeneous = np.column_stack([roots, np.ones(len(roots))])
    clip = homogeneous @ matrix.T
    uv = clip[:, :2] / (clip[:, 3:4] + 1e-12)
    px = (uv[:, 0] + 1.0) * 0.5 * max(width - 1, 1)
    py = (uv[:, 1] + 1.0) * 0.5 * max(height - 1, 1)
    distance = np.hypot(px - center_x, py - float(active_y_min))
    selected = (
        (px >= 0.0) & (px <= width - 1)
        & (py >= float(active_y_min) - radius)
        & (py < float(active_y_min))
        & (distance <= radius)
    )
    return selected


def reseed_roots_along_parting_boundaries(
    roots,
    head_mesh,
    calib,
    parting_mask,
    band_radius_px=12.0,
    boundary_offset_px=2.0,
    surface_distance=0.001,
    max_shift_distance=0.03,
):
    """把发缝带内的均匀发根局部移到左右边界并吸附头皮。"""
    roots = np.asarray(roots, dtype=np.float64).reshape(-1, 3)
    out = roots.copy()
    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    mask = np.asarray(parting_mask, dtype=bool)
    height, width = mask.shape
    mask_y, mask_x = np.where(mask)
    selected = np.zeros(len(roots), dtype=bool)
    sides = np.zeros(len(roots), dtype=np.int8)
    if len(mask_x) == 0 or len(roots) == 0:
        return out.astype(np.float32), selected, {"reseeded_roots": 0}

    left_edge = np.full(height, np.nan, dtype=np.float64)
    right_edge = np.full(height, np.nan, dtype=np.float64)
    for row in np.unique(mask_y):
        row_x = mask_x[mask_y == row]
        left_edge[row] = float(np.min(row_x))
        right_edge[row] = float(np.max(row_x))
    valid_rows = np.flatnonzero(np.isfinite(left_edge))

    homogeneous = np.column_stack([roots, np.ones(len(roots))])
    clip = homogeneous @ matrix.T
    uv = clip[:, :2] / (clip[:, 3:4] + 1e-12)
    px = (uv[:, 0] + 1.0) * 0.5 * (width - 1)
    py = (uv[:, 1] + 1.0) * 0.5 * (height - 1)
    rows = np.rint(py).astype(np.int64).clip(0, height - 1)
    row_valid = np.isfinite(left_edge[rows])
    center = 0.5 * (left_edge[rows] + right_edge[rows])
    signed = px - center
    inside_image = (
        (px >= 0.0) & (px <= width - 1)
        & (py >= valid_rows[0]) & (py <= valid_rows[-1])
    )
    selected = (
        inside_image
        & row_valid
        & (np.abs(signed) <= float(band_radius_px))
    )
    ambiguous = selected & (np.abs(signed) < 0.5)
    sides[selected] = np.where(signed[selected] < 0.0, -1, 1)
    ambiguous_ids = np.flatnonzero(ambiguous)
    sides[ambiguous_ids] = np.where(ambiguous_ids % 2 == 0, -1, 1)
    selected_ids = np.flatnonzero(selected)
    if len(selected_ids) == 0:
        return out.astype(np.float32), selected, {"reseeded_roots": 0}

    target_px = px[selected_ids].copy()
    left_ids = sides[selected_ids] < 0
    target_px[left_ids] = (
        left_edge[rows[selected_ids[left_ids]]] - float(boundary_offset_px)
    )
    target_px[~left_ids] = (
        right_edge[rows[selected_ids[~left_ids]]] + float(boundary_offset_px)
    )
    target_uv = np.column_stack([
        target_px / max(width - 1, 1) * 2.0 - 1.0,
        py[selected_ids] / max(height - 1, 1) * 2.0 - 1.0,
    ])
    inverse = np.linalg.inv(matrix)
    # The front matrix intentionally combines dense image x/y alignment with
    # the legacy neural-depth row. It is a valid projection but not a physical
    # camera for ray casting. Preserve each root's clip depth/w, change only
    # image x, then perform a local closest-surface query from that provisional
    # point. This prevents a ray from hitting the far side of the head.
    selected_clip = clip[selected_ids].copy()
    selected_clip[:, 0] = target_uv[:, 0] * selected_clip[:, 3]
    selected_clip[:, 1] = target_uv[:, 1] * selected_clip[:, 3]
    provisional_h = selected_clip @ inverse.T
    provisional = provisional_h[:, :3] / (provisional_h[:, 3:4] + 1e-12)
    legacy_mesh = (
        head_mesh
        if isinstance(head_mesh, o3d.geometry.TriangleMesh)
        else o3d.io.read_triangle_mesh(str(head_mesh))
    )
    if not legacy_mesh.has_vertices() or not legacy_mesh.has_triangles():
        raise ValueError("发根重播种需要有效的头皮三角网格")
    import trimesh
    proximity_mesh = trimesh.Trimesh(
        vertices=np.asarray(legacy_mesh.vertices),
        faces=np.asarray(legacy_mesh.triangles),
        process=False,
    )
    points, _, face_ids = trimesh.proximity.closest_point(
        proximity_mesh, provisional
    )
    normals = proximity_mesh.face_normals[np.asarray(face_ids, dtype=np.int64)]
    shift = np.linalg.norm(points - roots[selected_ids], axis=1)
    valid_hit = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(shift)
        & (shift <= float(max_shift_distance))
    )
    center_3d = np.asarray(legacy_mesh.get_center(), dtype=np.float64)
    points_valid = points[valid_hit]
    hit_normals = normals[valid_hit].copy()
    inward = np.sum(
        hit_normals * (points_valid - center_3d), axis=1
    ) < 0.0
    hit_normals[inward] *= -1.0
    hit_normals /= np.linalg.norm(hit_normals, axis=1, keepdims=True) + 1e-12
    out[selected_ids[valid_hit]] = (
        points_valid + hit_normals * float(surface_distance)
    )
    selected[selected_ids[~valid_hit]] = False
    sides[selected_ids[~valid_hit]] = 0
    report = {
        "candidate_roots": int(len(selected_ids)),
        "reseeded_roots": int(np.sum(valid_hit)),
        "left_roots": int(np.sum(sides == -1)),
        "right_roots": int(np.sum(sides == 1)),
        "band_radius_px": float(band_radius_px),
        "boundary_offset_px": float(boundary_offset_px),
        "surface_distance_m": float(surface_distance),
        "max_shift_distance_m": float(max_shift_distance),
        "rejected_large_shift": int(np.sum(~valid_hit)),
        "accepted_shift_mm": {
            "median": float(np.median(shift[valid_hit]) * 1000.0)
            if np.any(valid_hit) else 0.0,
            "max": float(np.max(shift[valid_hit]) * 1000.0)
            if np.any(valid_hit) else 0.0,
        },
    }
    return out.astype(np.float32), selected, report


def trim_strands_by_forbidden_projection(
    strands, forbidden_mask, calib, trim_points=2, min_root_points=8
):
    """回缩进入 front 内部禁区的末端，避免硬停止留下尖刺。"""
    if forbidden_mask is None or trim_points <= 0:
        return strands, 0
    out = np.asarray(strands, dtype=np.float32).copy()
    mask = np.asarray(forbidden_mask) > 0
    h, w = mask.shape
    calib = np.asarray(calib[0] if isinstance(calib, (tuple, list)) else calib)
    changed = 0
    for i in range(out.shape[0]):
        pts = out[i]
        homo = np.concatenate(
            [pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1
        )
        clip = homo @ calib.T
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
        px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (w - 1)).astype(np.int64)
        py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (h - 1)).astype(np.int64)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        hit = np.zeros(len(pts), dtype=bool)
        hit[inside] = mask[py[inside], px[inside]]
        ids = np.where(hit)[0]
        if len(ids) == 0:
            continue
        first = int(ids[0])
        # 发束刚离开根部就撞上发缝时，它通常是错误的跨缝根，而不是
        # 应该被截断的正常发束。保留这类束会产生“从空中冒出来”的短段。
        if first < int(min_root_points):
            out[i, 1:] = out[i, 0]
            changed += 1
            continue
        keep = max(1, first - int(trim_points))
        if keep < len(pts) - 1:
            out[i, keep + 1:] = out[i, keep]
            changed += 1
    return out, changed


def trim_strands_crossing_parting(
    strands,
    parting_mask,
    calib,
    trim_points=2,
    drop_reversed_tips=False,
    head_mesh_path=None,
    strand_root_visibility=None,
    back_trim_px=0,
    drop_floating_arches=False,
    floating_arch_clearance=0.01,
    floating_arch_steps=100,
):
    """按发缝中心线左右分区，只截断真正穿越中心线的发束。"""
    mask = np.asarray(parting_mask) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return strands, 0
    h, w = mask.shape
    center = np.full(h, np.nan, dtype=np.float32)
    for y in np.unique(ys):
        center[y] = float(np.median(xs[ys == y]))
    valid = np.where(np.isfinite(center))[0]
    center[: valid[0]] = center[valid[0]]
    center[valid[-1] + 1 :] = center[valid[-1]]
    missing = ~np.isfinite(center)
    center[missing] = np.interp(np.where(missing)[0], valid, center[valid])
    # Do not extrapolate the detected front parting over the whole projection.
    # The top of the front image corresponds to the rearward crown here; a
    # small positive trim prevents the 2D centerline from carving a corridor
    # all the way towards the back of the head.
    active_y_min = min(
        int(valid[-1]), int(valid[0]) + max(0, int(back_trim_px))
    )
    active_y_max = int(valid[-1])
    root_visibility = None
    if strand_root_visibility is not None:
        root_visibility = np.asarray(strand_root_visibility, dtype=bool).reshape(-1)
        if len(root_visibility) != len(strands):
            raise ValueError(
                "strand_root_visibility must match the strand count: "
                f"{len(root_visibility)} != {len(strands)}"
            )

    calib = np.asarray(calib[0] if isinstance(calib, (tuple, list)) else calib)
    out = np.asarray(strands, dtype=np.float32).copy()
    changed = 0
    for i, pts in enumerate(out):
        if root_visibility is not None and not root_visibility[i]:
            continue
        homo = np.concatenate(
            [pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1
        )
        clip = homo @ calib.T
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
        px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (w - 1)).astype(np.int64)
        py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (h - 1)).astype(np.int64)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not inside[0]:
            continue
        if py[0] < active_y_min or py[0] > active_y_max:
            continue
        root_side = float(px[0] - center[np.clip(py[0], 0, h - 1)])
        if abs(root_side) < 1.0:
            continue
        side = np.zeros(len(pts), dtype=np.float32)
        valid_pts = inside & (py >= active_y_min) & (py <= active_y_max)
        side[valid_pts] = px[valid_pts] - center[np.clip(py[valid_pts], 0, h - 1)]
        crossed = valid_pts & (side * root_side < -1.0)
        ids = np.where(crossed)[0]
        if len(ids) == 0:
            continue
        first = int(ids[0])
        keep = max(1, first - int(trim_points))
        out[i, keep + 1 :] = out[i, keep]
        changed += 1

    # A line field can still curve back after a correctly oriented root step.
    # Those strands do not cross the centerline, but their tips accumulate on
    # it and look like floating roots. Remove only this narrow, reversed-tip
    # population; legitimate roots adjacent to the parting remain untouched.
    if not drop_reversed_tips:
        return out, changed

    segment_lengths = np.linalg.norm(out[:, 1:] - out[:, :-1], axis=2)
    moving = segment_lengths > 1e-7
    effective_last = np.where(
        moving, np.arange(1, out.shape[1])[None, :], 0
    ).max(axis=1)
    tips = out[np.arange(len(out)), effective_last]

    def project_pixels(points):
        homo = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        )
        clip = homo @ calib.T
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
        return np.column_stack([
            (ndc[:, 0] + 1.0) * 0.5 * (w - 1),
            (ndc[:, 1] + 1.0) * 0.5 * (h - 1),
        ])

    root_px = project_pixels(out[:, 0])
    tip_px = project_pixels(tips)
    root_y = np.rint(root_px[:, 1]).astype(np.int64).clip(0, h - 1)
    tip_y = np.rint(tip_px[:, 1]).astype(np.int64).clip(0, h - 1)
    root_side = root_px[:, 0] - center[root_y]
    tip_side = tip_px[:, 0] - center[tip_y]
    approaches_parting = (
        np.abs(tip_side) + 12.0 <= np.abs(root_side)
    )
    reversed_tip = (
        (np.abs(root_side) >= 12.0)
        & (np.abs(root_side) <= 96.0)
        & (np.abs(tip_side) <= 48.0)
        & approaches_parting
        & (root_px[:, 1] >= float(active_y_min))
        & (root_px[:, 1] <= float(active_y_max))
        & (tip_px[:, 1] >= float(active_y_min - 12))
        & (tip_px[:, 1] <= float(active_y_max + 20))
    )
    if root_visibility is not None:
        reversed_tip &= root_visibility
    head_mesh = None
    if head_mesh_path is not None:
        import trimesh
        head_mesh = trimesh.load(head_mesh_path, process=False)
        _, tip_surface_distance, _ = trimesh.proximity.closest_point(
            head_mesh, tips.astype(np.float64)
        )
        # Free tips close to the scalp are ordinary short hairs. Only reject
        # tips that visibly terminate in air while flowing back to the parting.
        reversed_tip &= tip_surface_distance > 0.006
    reversed_ids = np.where(reversed_tip)[0]
    if len(reversed_ids):
        out[reversed_ids, 1:] = out[reversed_ids, :1]
        changed += int(len(reversed_ids))

    # Some valid roots grow into sparse, high arches over the crown and then
    # return towards the parting. Moving their samples creates a kink at the
    # constrained/free boundary, so reject only these topological outliers.
    arch_steps = min(max(0, int(floating_arch_steps)), out.shape[1])
    if drop_floating_arches and arch_steps > 1 and head_mesh is not None:
        # A crown flyaway can originate at an occluded rear/side root, so root
        # position and front visibility are not valid candidate filters. Audit
        # every non-collapsed strand, then localize the decision by the 2D
        # crown region and the 3D scalp clearance of its sampled points.
        candidate_ids = np.where(~reversed_tip)[0]
        if len(candidate_ids):
            root_segments = out[candidate_ids, :arch_steps]
            flat_segments = root_segments.reshape(-1, 3).astype(np.float64)
            distance_chunks = []
            proximity_chunk_size = 50000
            for start in range(0, len(flat_segments), proximity_chunk_size):
                stop = min(start + proximity_chunk_size, len(flat_segments))
                _, chunk_distances, _ = trimesh.proximity.closest_point(
                    head_mesh, flat_segments[start:stop]
                )
                distance_chunks.append(chunk_distances)
            distances = np.concatenate(distance_chunks)
            distances = distances.reshape(len(candidate_ids), arch_steps)
            segment_px = project_pixels(flat_segments).reshape(
                len(candidate_ids), arch_steps, 2
            )
            segment_y = np.rint(segment_px[:, :, 1]).astype(np.int64).clip(0, h - 1)
            segment_side = segment_px[:, :, 0] - center[segment_y]
            crown_height_min = float(head_mesh.bounds[1, 1]) - 0.02
            crown_region = (
                (segment_px[:, :, 1] >= float(valid[0] - 24))
                & (segment_px[:, :, 1] <= float(active_y_max + 24))
                & (np.abs(segment_side) <= 180.0)
                & (root_segments[:, :, 1] >= crown_height_min)
            )
            floating = np.any(
                crown_region & (distances > float(floating_arch_clearance)), axis=1
            )
            floating_ids = candidate_ids[floating]
            if len(floating_ids):
                out[floating_ids, 1:] = out[floating_ids, :1]
                changed += int(len(floating_ids))
    return out, changed


def attach_strand_roots_to_head(
    strands, sdf_vol, normal_vol, b_min_t, b_max_t, device,
    attach_steps=12, target_distance=0.008, strength=0.65
):
    """逐束把根部吸附到头皮，并保持根部发流方向。

    旧实现对前 ``attach_steps`` 个点只做一次独立的 SDF 修正：当根点
    偏离头皮较远时，修正量不够，而且相邻点被拉向不同的法向，容易形成
    ``悬空根 + 弧线``。这里先迭代求根点，再把同一个位移按距离衰减地
    传播到根部，最后只修正“向头皮内部”的第一段方向。
    """
    if attach_steps <= 0 or target_distance <= 0.0 or strength <= 0.0:
        return strands, 0
    out = np.asarray(strands, dtype=np.float32).copy()
    steps = min(int(attach_steps), out.shape[1])
    roots = torch.from_numpy(out[:, 0].T).float().to(device)
    corrected = 0
    # A few short Newton-like steps are important for roots that are several
    # voxels away from the head surface.  Use the signed distance directly so
    # both outside and penetrating roots converge to the same shell.
    for _ in range(4):
        sdf = query_grid(sdf_vol, roots, b_min_t, b_max_t).reshape(-1)
        normal = torch.nn.functional.normalize(
            query_grid(normal_vol, roots, b_min_t, b_max_t).reshape(3, -1), dim=0
        )
        delta = sdf - float(target_distance)
        active = torch.abs(delta) > 0.001
        roots = roots - float(strength) * delta.unsqueeze(0) * normal
        corrected += int(active.sum().item())

    original_roots = torch.from_numpy(out[:, 0].T).float().to(device)
    root_delta = roots - original_roots
    root_ramp = torch.linspace(1.0, 0.0, steps, device=device).view(1, steps, 1)
    out[:, :steps] += (
        root_delta.T[:, None, :].cpu().numpy() * root_ramp.cpu().numpy()
    )
    out[:, 0] = roots.T.cpu().numpy()

    # Do not allow the first segment to point into the scalp.  We retain its
    # observed tangential component and add only the minimum outward normal
    # component needed to leave the surface; this avoids the arcs caused by a
    # global scalp-tangent constraint.
    if steps >= 2:
        root_normal = torch.nn.functional.normalize(
            query_grid(normal_vol, roots, b_min_t, b_max_t), dim=0
        )
        p0 = torch.from_numpy(out[:, 0].T).float().to(device)
        p1 = torch.from_numpy(out[:, 1].T).float().to(device)
        segment = p1 - p0
        length = torch.norm(segment, dim=0, keepdim=True).clamp_min(1e-6)
        unit = segment / length
        normal_component = torch.sum(unit * root_normal, dim=0, keepdim=True)
        inward = normal_component < 0.0
        tangent = unit - normal_component * root_normal
        tangent = torch.nn.functional.normalize(tangent, dim=0)
        outward_unit = torch.nn.functional.normalize(
            tangent + 0.15 * root_normal, dim=0
        )
        safe_unit = torch.where(inward, outward_unit, unit)
        out[:, 1] = (p0 + safe_unit * length).T.cpu().numpy()
    return out, corrected


def project_roots_to_head_surface(
    roots, sdf_vol, normal_vol, b_min_t, b_max_t, device,
    target_distance=0.004, iterations=4
):
    """用头模 SDF/法向把 roots 投影到最近头皮表面附近。"""
    if iterations <= 0:
        return roots, 0
    points = roots.clone().reshape(3, -1)
    corrected = 0
    for _ in range(int(iterations)):
        sdf = query_grid(sdf_vol, points, b_min_t, b_max_t)
        normal = torch.nn.functional.normalize(
            query_grid(normal_vol, points, b_min_t, b_max_t), dim=0
        )
        delta = sdf - float(target_distance)
        active = torch.abs(delta) > 0.001
        points = points - delta.unsqueeze(0) * normal
        corrected += int(active.sum().item())
    return points, corrected


def project_strand_roots_to_mesh(
    strands, mesh_path, attach_steps=16, target_distance=0.003
):
    """用头模三角网格最近点做最终根点校正，避免 SDF 边界漂移。"""
    if attach_steps <= 0 or target_distance <= 0.0:
        return strands, 0
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return strands, 0
    out = np.asarray(strands, dtype=np.float32).copy()
    roots = out[:, 0].astype(np.float64)
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, roots)
    normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)]
    # Make the normal point from the surface toward the original root.
    inward = np.sum(normals * (roots - closest), axis=1) < 0.0
    normals[inward] *= -1.0
    projected = closest + normals * float(target_distance)
    delta = projected.astype(np.float32) - out[:, 0]
    steps = min(int(attach_steps), out.shape[1])
    ramp = np.linspace(1.0, 0.0, steps, dtype=np.float32)[None, :, None]
    out[:, :steps] += delta[:, None, :] * ramp
    out[:, 0] = projected.astype(np.float32)
    return out, int(np.sum(np.linalg.norm(delta, axis=1) > 0.001))


def attach_parting_root_segments_to_mesh(
    strands,
    parting_mask,
    calib,
    head_mesh,
    attach_steps=8,
    hard_steps=3,
    target_distance=0.0015,
    influence_radius_px=48.0,
    back_trim_px=0,
    strand_root_visibility=None,
):
    """让发缝邻域的根段从头皮表面连续长出，而不只约束第 0 点。

    前 ``hard_steps`` 个点严格投影到头皮外壳，其余根段在
    ``attach_steps`` 内逐渐恢复原始 PDE 轨迹。这样可以消除第 0 点贴头皮、
    第 1/2 点立即悬空造成的视觉断层，同时不改变发缝以外的发丝。
    """
    if attach_steps <= 0 or hard_steps <= 0 or target_distance <= 0.0:
        return strands, 0, 0
    mask = np.asarray(parting_mask) > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return strands, 0, 0

    out = np.asarray(strands, dtype=np.float32).copy()
    steps = min(int(attach_steps), out.shape[1])
    hard = min(int(hard_steps), steps)
    height, width = mask.shape
    center = np.full(height, np.nan, dtype=np.float32)
    for row in np.unique(ys):
        center[row] = float(np.median(xs[ys == row]))
    valid_rows = np.flatnonzero(np.isfinite(center))
    center = np.interp(
        np.arange(height), valid_rows, center[valid_rows]
    ).astype(np.float32)
    active_y_min = min(
        int(valid_rows[-1]),
        int(valid_rows[0]) + max(0, int(back_trim_px)),
    )
    active_y_max = int(valid_rows[-1])

    calib = np.asarray(calib[0] if isinstance(calib, (tuple, list)) else calib)
    roots = out[:, 0]
    homogeneous = np.concatenate(
        [roots, np.ones((len(roots), 1), dtype=np.float32)], axis=1
    )
    clip = homogeneous @ calib.T
    ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
    root_x = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
    root_y = (ndc[:, 1] + 1.0) * 0.5 * (height - 1)
    root_rows = np.rint(root_y).astype(np.int64).clip(0, height - 1)
    segment_lengths = np.linalg.norm(out[:, 1:] - out[:, :-1], axis=2)
    selected = (
        (root_y >= float(active_y_min))
        & (root_y <= float(active_y_max))
        & (np.abs(root_x - center[root_rows]) <= float(influence_radius_px))
        & np.any(segment_lengths > 1e-7, axis=1)
    )
    if strand_root_visibility is not None:
        visibility = np.asarray(strand_root_visibility, dtype=bool).reshape(-1)
        if len(visibility) != len(out):
            raise ValueError(
                "strand_root_visibility must match the strand count: "
                f"{len(visibility)} != {len(out)}"
            )
        selected &= visibility
    selected_ids = np.flatnonzero(selected)
    if len(selected_ids) == 0:
        return out, 0, 0

    import trimesh

    mesh = (
        head_mesh
        if isinstance(head_mesh, trimesh.Trimesh)
        else trimesh.load(head_mesh, process=False)
    )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return out, 0, 0
    original = out[selected_ids, :steps].astype(np.float64)
    flat = original.reshape(-1, 3)
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, flat)
    normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)].copy()
    inward = np.sum(normals * (flat - closest), axis=1) < 0.0
    normals[inward] *= -1.0
    shell = (closest + normals * float(target_distance)).reshape(original.shape)

    weights = np.ones(steps, dtype=np.float32)
    if steps > hard:
        weights[hard:] = np.linspace(
            1.0, 0.0, steps - hard + 1, dtype=np.float32
        )[1:]
    corrected = (
        original * (1.0 - weights[None, :, None])
        + shell * weights[None, :, None]
    )
    out[selected_ids, :steps] = corrected.astype(np.float32)
    return out, int(len(selected_ids)), int(len(selected_ids) * hard)


def taper_front_parting_domain_mask(mask, back_taper_px=0):
    """让 PDE 发缝挖空域在后冠端逐行收窄并自然闭合。

    完整发缝 mask 仍用于方向护栏和最终发根安全检查；这里只生成供
    segmentation、silhouette guard 与 3D corridor 使用的求解域负掩码。
    图像上方（最小 y）视为发缝后冠端。
    """
    base = np.asarray(mask) > 0
    taper = max(0, int(back_taper_px))
    if taper == 0 or not np.any(base):
        return base.copy()

    ys = np.flatnonzero(np.any(base, axis=1))
    y_min = int(ys[0])
    out = base.copy()
    for y in ys:
        distance = float(int(y) - y_min)
        if distance >= taper:
            continue
        row_x = np.flatnonzero(base[y])
        if len(row_x) == 0:
            continue
        linear = np.clip(distance / float(taper), 0.0, 1.0)
        weight = linear * linear * (3.0 - 2.0 * linear)
        if weight <= 0.0:
            out[y] = False
            continue
        center = 0.5 * float(row_x[0] + row_x[-1])
        half_width = 0.5 * float(row_x[-1] - row_x[0] + 1) * weight
        keep = (
            np.abs(np.arange(base.shape[1], dtype=np.float32) - center)
            < half_width
        )
        out[y] &= keep
    return out


def expand_front_parting_mask(mask, image_dilation_px=3, top_extension_px=16):
    """膨胀发缝并沿已检测曲线向头顶延伸，避免顶部跨缝。"""
    import cv2
    base = np.asarray(mask) > 0
    radius = max(0, int(image_dilation_px))
    if radius:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        base = cv2.dilate(base.astype(np.uint8), kernel, iterations=1) > 0
    extension = max(0, int(top_extension_px))
    if extension:
        extended = base.copy()
        for shift in range(1, extension + 1):
            extended[:-shift] |= base[shift:]
        base = extended
    return base


def build_front_parting_corridor(
    forbidden_mask, calib, b_min, b_max, resolution, domain_mask,
    image_dilation_px=3, voxel_dilation=1, top_extension_px=16,
    surface_mesh=None, surface_max_distance=0.0,
):
    """将 front 发缝负掩码投影为有厚度的 3D 禁止走廊。

    单像素投影只会删除一列体素，PDE 仍能在相邻像素/深度层绕过发缝。
    先在 front 图像中膨胀，再在体素域做小半径膨胀，形成连续的屏障。
    """
    if forbidden_mask is None:
        return np.zeros_like(domain_mask, dtype=bool)
    if np.isscalar(resolution):
        shape = (int(resolution), int(resolution), int(resolution))
    else:
        shape = tuple(int(x) for x in resolution)
    axes = [
        np.linspace(float(b_min[i]), float(b_max[i]), shape[i], dtype=np.float32)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    homo = np.concatenate(
        [grid, np.ones((len(grid), 1), dtype=np.float32)], axis=1
    )
    if isinstance(calib, (tuple, list)):
        calib = calib[0]
    clip = homo @ np.asarray(calib).T
    ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
    h, w = np.asarray(forbidden_mask).shape
    px = np.rint((ndc[:, 0] + 1.0) * 0.5 * (w - 1)).astype(np.int64)
    py = np.rint((ndc[:, 1] + 1.0) * 0.5 * (h - 1)).astype(np.int64)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    corridor = np.zeros(len(grid), dtype=bool)
    mask = expand_front_parting_mask(
        forbidden_mask,
        image_dilation_px=image_dilation_px,
        top_extension_px=top_extension_px,
    )
    corridor[inside] = mask[py[inside], px[inside]]
    corridor = corridor.reshape(shape)
    if int(voxel_dilation) > 0:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        corridor = binary_dilation(
            corridor,
            structure=generate_binary_structure(3, 2),
            iterations=int(voxel_dilation),
        )
    corridor = corridor & np.asarray(domain_mask, dtype=bool)
    max_distance = float(surface_max_distance)
    if surface_mesh is not None and max_distance > 0.0 and np.any(corridor):
        mesh = (
            surface_mesh
            if isinstance(surface_mesh, trimesh.Trimesh)
            else trimesh.load(surface_mesh, process=False)
        )
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("头皮薄切口需要有效的三角网格")
        candidate_ids = np.flatnonzero(corridor)
        candidate_points = grid[candidate_ids].astype(np.float64)
        distance_chunks = []
        for start in range(0, len(candidate_points), 50000):
            stop = min(start + 50000, len(candidate_points))
            _, distances, _ = trimesh.proximity.closest_point(
                mesh, candidate_points[start:stop]
            )
            distance_chunks.append(distances)
        distances = np.concatenate(distance_chunks)
        keep = distances <= max_distance
        restricted = np.zeros(corridor.size, dtype=bool)
        restricted[candidate_ids[keep]] = True
        corridor = restricted.reshape(shape)
    return corridor


def build_front_parting_interface_constraints(
    forbidden_mask,
    calib,
    b_min,
    b_max,
    resolution,
    domain_mask,
    surface_mesh,
    surface_max_distance=0.012,
    back_taper_px=0,
):
    """构造贴近头皮的发缝内部滑移界面及其切平面法向。"""
    mask = taper_front_parting_domain_mask(
        forbidden_mask, back_taper_px=back_taper_px
    )
    interface = build_front_parting_corridor(
        mask,
        calib,
        b_min,
        b_max,
        resolution,
        domain_mask,
        image_dilation_px=0,
        voxel_dilation=0,
        top_extension_px=0,
        surface_mesh=surface_mesh,
        surface_max_distance=surface_max_distance,
    )
    normals = np.zeros((3, *interface.shape), dtype=np.float32)
    candidate_ids = np.flatnonzero(interface)
    if len(candidate_ids) == 0:
        return interface, normals

    mesh = (
        surface_mesh
        if isinstance(surface_mesh, trimesh.Trimesh)
        else trimesh.load(surface_mesh, process=False)
    )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("发缝内部界面需要有效的头皮三角网格")
    shape = interface.shape
    axes = [
        np.linspace(float(b_min[i]), float(b_max[i]), shape[i])
        for i in range(3)
    ]
    indices = np.column_stack(np.unravel_index(candidate_ids, shape))
    points = np.column_stack([
        axes[axis][indices[:, axis]] for axis in range(3)
    ])
    _, _, face_ids = trimesh.proximity.closest_point(mesh, points)
    scalp_normals = mesh.face_normals[np.asarray(face_ids, dtype=np.int64)]
    scalp_normals /= (
        np.linalg.norm(scalp_normals, axis=1, keepdims=True) + 1e-12
    )

    matrix = np.asarray(
        calib[0] if isinstance(calib, (tuple, list)) else calib,
        dtype=np.float64,
    )
    screen_x_gradient = matrix[0, :3]
    screen_x_gradient /= np.linalg.norm(screen_x_gradient) + 1e-12
    interface_normals = np.broadcast_to(
        screen_x_gradient, scalp_normals.shape
    ).copy()
    interface_normals -= (
        np.sum(interface_normals * scalp_normals, axis=1, keepdims=True)
        * scalp_normals
    )
    interface_normals /= (
        np.linalg.norm(interface_normals, axis=1, keepdims=True) + 1e-12
    )
    normals.reshape(3, -1)[:, candidate_ids] = interface_normals.T.astype(
        np.float32
    )
    return interface, normals


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
    parser.add_argument(
        "--front_parting_mask",
        default=None,
        help=(
            "Optional front internal negative mask. Pixels set in this mask "
            "are removed from front seg/root guidance (experimental)."
        ),
    )
    parser.add_argument(
        "--front_parting_trim_points",
        type=int,
        default=2,
        help="回缩进入 front 发缝禁区的末端采样点数",
    )
    parser.add_argument(
        "--front_parting_3d_corridor", action="store_true",
        help="将 front 发缝禁区沿相机射线扣除 PDE 三维域（实验）",
    )
    parser.add_argument(
        "--front_parting_corridor_px", type=int, default=3,
        help="发缝 2D 禁区膨胀半径（像素）",
    )
    parser.add_argument(
        "--front_parting_corridor_voxels", type=int, default=1,
        help="发缝 3D 走廊膨胀半径（体素）",
    )
    parser.add_argument(
        "--front_parting_corridor_surface_height",
        type=float,
        default=0.0,
        help=(
            "仅删除距头皮网格不超过该高度的发缝体素（米）；"
            "0 保留贯穿深度的旧走廊"
        ),
    )
    parser.add_argument(
        "--front_parting_pde_interface",
        action="store_true",
        help="将发缝作为 PDE 域内无穿透滑移界面，而不是挖空求解域",
    )
    parser.add_argument(
        "--front_parting_interface_surface_height",
        type=float,
        default=0.012,
        help="PDE 发缝界面距头皮的最大高度（米）",
    )
    parser.add_argument(
        "--front_parting_interface_confidence", type=float, default=0.5,
        help="PDE 发缝界面的法向零通量约束强度",
    )
    parser.add_argument(
        "--front_parting_interface_length", type=float, default=0.006,
        help="PDE 发缝界面约束的物理作用长度（米）",
    )
    parser.add_argument(
        "--front_parting_interface_back_taper_px", type=int, default=12,
        help="PDE 发缝界面在后冠端渐闭的长度（像素）",
    )
    parser.add_argument(
        "--front_parting_top_extension_px", type=int, default=16,
        help="发缝沿检测曲线向头顶延伸的像素数",
    )
    parser.add_argument(
        "--front_parting_root_sign_radius_px", type=float, default=96.0,
        help="按发缝左右侧校正 RK4 初始方向符号的最大像素距离",
    )
    parser.add_argument(
        "--front_parting_root_sign_steps", type=int, default=0,
        help="根部保持远离发缝方向的 RK4 采样步数",
    )
    parser.add_argument(
        "--front_parting_scalp_geodesic_reference",
        action="store_true",
        help="用提升到三维头皮的发缝曲线替代二维图像根方向参考",
    )
    parser.add_argument(
        "--front_parting_geodesic_radius_px", type=float, default=48.0,
        help="三维头皮发缝切向根参考的图像影响半径",
    )
    parser.add_argument(
        "--front_parting_geodesic_endpoint_blend_px", type=float, default=18.0,
        help="后冠端从左右外向切线混合到共同后向切线的长度",
    )
    parser.add_argument(
        "--front_parting_geodesic_curve_neighbors", type=int, default=8,
        help="每个发缝行拟合三维头皮曲线使用的邻近根样本数",
    )
    parser.add_argument(
        "--front_parting_geodesic_curve_mode",
        choices=("root_fit", "mesh_raycast", "hybrid_raycast_cap"),
        default="root_fit",
        help="三维发缝曲线来源：邻近发根拟合或前视射线命中可见头皮",
    )
    parser.add_argument(
        "--front_parting_root_direction_min_component", type=float, default=0.15,
        help="根方向参考在 RK4 合成方向中的最小分量",
    )
    parser.add_argument(
        "--front_parting_direction_barrier_radius_px", type=float, default=0.0,
        help="发缝中心线两侧启用动态速度屏障的半径；0=关闭",
    )
    parser.add_argument(
        "--front_parting_direction_barrier_min_component",
        type=float,
        default=0.05,
        help="动态速度屏障要求的最小向外方向分量",
    )
    parser.add_argument(
        "--front_parting_direction_barrier_margin_px",
        type=float,
        default=1.0,
        help="RK4 候选点相对左右发缝边界保留的最小外侧距离",
    )
    parser.add_argument(
        "--front_parting_direction_barrier_back_taper_px",
        type=float,
        default=0.0,
        help="发缝后冠端从闭合到完整边界约束的平滑渐隐长度",
    )
    parser.add_argument(
        "--front_parting_direction_barrier_back_cap_radius_px",
        type=float,
        default=0.0,
        help="发缝后冠端圆角方向护栏半径；0=关闭",
    )
    parser.add_argument(
        "--front_parting_domain_back_taper_px",
        type=int,
        default=0,
        help="PDE 二值发缝挖空域在后冠端逐行收窄闭合的长度；0=关闭",
    )
    parser.add_argument(
        "--front_parting_drop_reversed_tips", action="store_true",
        help="删除根点远离发缝但末端堆积在发缝上的反向发束",
    )
    parser.add_argument(
        "--front_parting_back_trim_px", type=int, default=0,
        help="从 front 发缝的后脑端缩短约束范围（像素）",
    )
    parser.add_argument(
        "--front_parting_drop_floating_arches", action="store_true",
        help="删除发缝冠部根段净空异常的漂浮拱弧",
    )
    parser.add_argument(
        "--front_parting_floating_arch_clearance", type=float, default=0.01,
        help="判定发缝漂浮拱弧的头皮净空阈值（米）",
    )
    parser.add_argument(
        "--front_parting_floating_arch_steps", type=int, default=100,
        help="检查漂浮拱弧的根段采样点数",
    )
    parser.add_argument(
        "--front_parting_surface_attach_steps", type=int, default=8,
        help="发缝邻域贴头皮过渡的根段采样点数；0=关闭",
    )
    parser.add_argument(
        "--front_parting_surface_hard_steps", type=int, default=3,
        help="发缝邻域严格投影到头皮外壳的根段采样点数",
    )
    parser.add_argument(
        "--front_parting_surface_radius_px", type=float, default=48.0,
        help="发缝中心线两侧启用根段贴附的最大像素距离",
    )
    parser.add_argument(
        "--front_parting_surface_distance", type=float, default=0.001,
        help="发缝邻域根段相对头皮表面的目标距离（米）",
    )
    parser.add_argument(
        "--front_parting_reseed_roots", action="store_true",
        help="把发缝带内固定发根重定位到 DINO 发缝左右边界",
    )
    parser.add_argument(
        "--front_parting_reseed_radius_px", type=float, default=12.0,
        help="参与发缝边界重播种的中心线半径（像素）",
    )
    parser.add_argument(
        "--front_parting_reseed_offset_px", type=float, default=2.0,
        help="重播种发根相对发缝左右边界的外移距离（像素）",
    )
    parser.add_argument(
        "--front_parting_reseed_surface_distance", type=float, default=0.001,
        help="重播种根点相对头皮表面的外侧距离（米）",
    )
    parser.add_argument("--roots",
                        default="data/roots10k.obj")
    parser.add_argument(
        "--root_attach_steps", type=int, default=0,
        help="根部头皮吸附的采样点数；0=关闭",
    )
    parser.add_argument(
        "--root_attach_distance", type=float, default=0.008,
        help="根部允许离头皮的目标距离（米）",
    )
    parser.add_argument(
        "--root_attach_strength", type=float, default=0.65,
        help="根部头皮吸附强度",
    )
    parser.add_argument(
        "--rk4_root_collision_ramp_steps",
        type=int,
        default=0,
        help="RK4 根段碰撞壳由贴皮距离平滑增长到 5mm 的步数；0=旧行为",
    )
    parser.add_argument(
        "--rk4_root_collision_start_distance",
        type=float,
        default=0.0015,
        help="RK4 根段碰撞渐变起始距离（米）",
    )
    parser.add_argument(
        "--project_roots_to_head", action="store_true",
        help="将每个 root 投影到最近头皮表面（实验）",
    )
    parser.add_argument(
        "--root_surface_distance", type=float, default=0.004,
        help="root 投影到头皮外侧的目标距离（米）",
    )
    parser.add_argument(
        "--root_projection_iterations", type=int, default=4,
        help="root 头皮投影迭代次数",
    )
    parser.add_argument(
        "--root_tangent_steps", type=int, default=0,
        help="root 出发后沿头皮切向约束的步数；0=关闭",
    )
    parser.add_argument(
        "--root_tangent_strength", type=float, default=0.8,
        help="root 初始切向约束强度",
    )
    parser.add_argument(
        "--front_parting_surface_follow_steps", type=int, default=0,
        help="重播种发根沿头皮等距壳生长的 RK4 步数；0=关闭",
    )
    parser.add_argument(
        "--front_parting_surface_follow_distance", type=float, default=0.0015,
        help="发缝根部曲面前导段离头皮距离（米）",
    )
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
    parser.add_argument(
        "--use_visible_hair_proxy",
        action="store_true",
        help=(
            "Build the PDE domain from Pixal3D triangles first hit through "
            "the multiview hair masks instead of the complete face/bust mesh"
        ),
    )
    parser.add_argument(
        "--hair_proxy_face_dilation",
        type=int,
        default=1,
        help="Topological face rings added around visible hair triangles",
    )
    parser.add_argument(
        "--surface_front_tolerance",
        type=float,
        default=0.01,
        help="Allowed volume in front of the visible hair surface, in meters",
    )
    parser.add_argument(
        "--surface_back_tolerance",
        type=float,
        default=0.06,
        help="Allowed volume behind the visible hair surface, in meters",
    )
    parser.add_argument(
        "--support_multiview_union",
        action="store_true",
        help="Let side/back hair evidence recover volume vetoed by front background",
    )
    parser.add_argument(
        "--surface_attraction_weight",
        type=float,
        default=0.0,
        help=(
            "Weak RK4 attraction toward the nearest visible hair-proxy "
            "surface; 0 disables it (default)"
        ),
    )
    parser.add_argument(
        "--surface_attraction_deadzone",
        type=float,
        default=0.015,
        help="No surface attraction within this distance in meters",
    )
    parser.add_argument(
        "--surface_attraction_full_distance",
        type=float,
        default=0.05,
        help="Distance in meters where surface attraction reaches full weight",
    )
    parser.add_argument(
        "--front_depth_bias",
        type=float,
        default=0.0,
        help=(
            "Optional RK4 direction component along the front camera ray; "
            "0 disables it (default)"
        ),
    )
    parser.add_argument(
        "--root_layer_depth_bias",
        type=float,
        default=0.0,
        help=(
            "Maximum signed front-ray bias assigned from each scalp root's "
            "robust depth layer; 0 disables it (default)"
        ),
    )
    parser.add_argument(
        "--root_layer_depth_offset",
        type=float,
        default=0.0,
        help=(
            "Maximum signed final displacement in meters assigned from each "
            "root depth layer; 0 disables it (default)"
        ),
    )
    parser.add_argument(
        "--root_layer_cluster_scale_px",
        type=float,
        default=64.0,
        help=(
            "Depth-layer separation used as a third KMeans feature in pixel "
            "units when root-layer depth is enabled"
        ),
    )
    parser.add_argument("--pde_cg_tol", type=float, default=1e-4)
    parser.add_argument("--pde_cg_maxiter", type=int, default=2000)
    parser.add_argument("--pde_anisotropy", type=float, default=0.8)
    parser.add_argument(
        "--pde_solver_mode",
        choices=("legacy_smooth", "screened_poisson"),
        default="legacy_smooth",
        help=(
            "Multi-view field solver. legacy_smooth is the retained EDT + "
            "Gaussian baseline; screened_poisson runs the true masked PDE"
        ),
    )
    parser.add_argument(
        "--pde_side_soft_confidence",
        type=float,
        default=0.10,
        help="Relative confidence of accepted synthesized side/back directions",
    )
    parser.add_argument(
        "--pde_side_screening_length",
        type=float,
        default=0.02,
        help="Metric screening length for side/back soft PDE sources",
    )
    parser.add_argument(
        "--pde_side_max_angle_degrees",
        type=float,
        default=60.0,
        help="Reject side/back directions farther from the trusted line field",
    )
    parser.add_argument(
        "--pde_side_constraint_mode",
        choices=("soft", "hard"),
        default="soft",
        help="Use accepted synthesized side/back directions as soft or hard PDE data",
    )
    parser.add_argument(
        "--pde_domain_tangent_confidence",
        type=float,
        default=0.0,
        help=(
            "Strength of the PDE outer-boundary tangency energy; "
            "0 disables it (default)"
        ),
    )
    parser.add_argument(
        "--pde_domain_tangent_length",
        type=float,
        default=0.01,
        help="Metric screening length for PDE domain tangency",
    )
    parser.add_argument(
        "--pde_domain_padding_voxels",
        type=int,
        default=0,
        help=(
            "Explicit voxel padding of a fused multiview PDE domain; "
            "unlike legacy dilation this also applies when hair_volume exists"
        ),
    )
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
        "--rk4_domain_guard",
        action="store_true",
        help=(
            "Use the PDE-domain signed distance to remove outward field "
            "components and project escaped RK4 candidates back inside"
        ),
    )
    parser.add_argument(
        "--rk4_domain_guard_margin",
        type=float,
        default=0.0,
        help=(
            "Boundary tangent zone in meters; <=0 selects two maximum-axis "
            "voxels automatically"
        ),
    )
    parser.add_argument(
        "--rk4_domain_projection_epsilon",
        type=float,
        default=0.001,
        help="Inward offset in meters after projecting an escaped candidate",
    )
    parser.add_argument(
        "--rk4_local_domain_recovery",
        action="store_true",
        help="Recover low-field RK4 samples using local PDE-domain SDF continuation",
    )
    parser.add_argument(
        "--rk4_local_domain_recovery_margin",
        type=float,
        default=0.0,
        help="Interior continuation band in meters; <=0 reuses the domain guard margin",
    )
    parser.add_argument(
        "--rk4_local_domain_recovery_max_angle_degrees",
        type=float,
        default=45.0,
        help="Maximum line-field angle for accepting local continuation",
    )
    parser.add_argument(
        "--rk4_domain_inward_bias",
        type=float,
        default=0.05,
        help=(
            "Relative inward barrier after removing an outward boundary "
            "component (default: 0.05)"
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

    front_parting_mask = None
    front_parting_domain_mask = None
    if args.front_parting_mask:
        if "front" not in seg_masks:
            raise ValueError("--front_parting_mask requires a front view")
        parting = imageio.imread(args.front_parting_mask)
        if parting.ndim == 3:
            parting = parting[:, :, 0]
        if parting.shape != seg_masks["front"].shape:
            parting = cv2.resize(
                parting.astype(np.uint8),
                (seg_masks["front"].shape[1], seg_masks["front"].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        parting = parting > 127
        front_parting_mask = parting
        front_parting_domain_mask = taper_front_parting_domain_mask(
            parting,
            back_taper_px=args.front_parting_domain_back_taper_px,
        )
        seg_masks["front"] &= ~front_parting_domain_mask
        imageio.imwrite(
            os.path.join(out_dir, "front_parting_applied_seg.png"),
            (seg_masks["front"] * 255).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(out_dir, "front_parting_domain_mask.png"),
            (front_parting_domain_mask * 255).astype(np.uint8),
        )
        print(
            "  Front parting exclusion: "
            f"full={int(parting.sum())}, "
            f"domain={int(front_parting_domain_mask.sum())} pixels, "
            f"back_taper={args.front_parting_domain_back_taper_px}px"
        )

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
    domain_surface_path = aligned_surface_path
    if args.use_visible_hair_proxy:
        domain_surface_path = os.path.join(out_dir, "visible_hair_proxy.obj")
        build_visible_hair_proxy_mesh(
            aligned_surface_path,
            geometry_calibs,
            seg_masks,
            domain_surface_path,
            face_dilation_rings=args.hair_proxy_face_dilation,
        )
        print(f"  Hair-only PDE surface: {domain_surface_path}")
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
            forbidden_masks=(
                {"front": front_parting_domain_mask}
                if front_parting_domain_mask is not None else None
            ),
        )
        print(
            f"  {silhouette_guard.summary()}: RK4 growth is silhouette-constrained "
            f"(grace={args.silhouette_grace_steps} steps)"
        )
    else:
        print("  Silhouette guard DISABLED (--disable_silhouette_guard)")

    hair_volume, raw_surface_shell, shell_counts = build_visible_mesh_surface_shell(
        domain_surface_path,
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
                domain_surface_path,
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
        occluder_mesh_path=domain_surface_path,
        visibility_tolerance=args.head_occlusion_tolerance,
        surface_front_tolerance=(
            args.surface_front_tolerance if args.use_visible_hair_proxy else None
        ),
        surface_back_tolerance=(
            args.surface_back_tolerance if args.use_visible_hair_proxy else None
        ),
        primary_authoritative=not args.support_multiview_union,
    )
    boundary_mask &= seg_support_volume
    fused_orien[:, ~boundary_mask] = 0.0
    view_ownership[~boundary_mask] = -1
    hair_volume = build_mesh_metric_band(
        domain_surface_path,
        seg_support_volume,
        b_min,
        b_max,
        band_width=args.pde_band_width,
    )
    hair_volume |= boundary_mask
    if args.front_parting_3d_corridor and front_parting_domain_mask is not None:
        parting_corridor = build_front_parting_corridor(
            front_parting_domain_mask,
            geometry_calibs["front"],
            b_min,
            b_max,
            args.pde_resolution,
            hair_volume,
            image_dilation_px=args.front_parting_corridor_px,
            voxel_dilation=args.front_parting_corridor_voxels,
            top_extension_px=args.front_parting_top_extension_px,
            surface_mesh=(
                domain_surface_path
                if args.front_parting_corridor_surface_height > 0.0 else None
            ),
            surface_max_distance=(
                args.front_parting_corridor_surface_height
            ),
        )
        hair_volume &= ~parting_corridor
        boundary_mask &= ~parting_corridor
        fused_orien[:, parting_corridor] = 0.0
        view_ownership[parting_corridor] = -1
        print(
            "  Front parting 3D corridor: "
            f"removed={int(parting_corridor.sum())} voxels, "
            "surface_height="
            f"{args.front_parting_corridor_surface_height:.4f}m"
        )
    surface_attraction_np = None
    if args.surface_attraction_weight > 0.0:
        if not args.use_visible_hair_proxy:
            raise ValueError(
                "--surface_attraction_weight requires --use_visible_hair_proxy"
            )
        surface_attraction_np = build_surface_attraction_volume(
            raw_surface_shell,
            b_min,
            b_max,
            domain_mask=hair_volume,
        )
    print(
        f"  Visible outer shell: raw={raw_surface_shell.sum()}, "
        f"metric_band={hair_volume.sum()}, per_view={shell_counts}; "
        f"fused contribution voxels={boundary_mask.sum()}"
    )

    parting_interface_np = None
    parting_interface_normals_np = None
    if args.front_parting_pde_interface:
        if front_parting_mask is None:
            raise ValueError(
                "--front_parting_pde_interface 需要 --front_parting_mask"
            )
        parting_interface_np, parting_interface_normals_np = (
            build_front_parting_interface_constraints(
                front_parting_mask,
                geometry_calibs["front"],
                b_min,
                b_max,
                args.pde_resolution,
                hair_volume,
                domain_surface_path,
                surface_max_distance=(
                    args.front_parting_interface_surface_height
                ),
                back_taper_px=(
                    args.front_parting_interface_back_taper_px
                ),
            )
        )
        print(
            "  Front parting PDE interface: "
            f"voxels={int(parting_interface_np.sum())}, "
            f"height={args.front_parting_interface_surface_height:.4f}m, "
            f"confidence={args.front_parting_interface_confidence:.3f}"
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
        parting_interface=(
            parting_interface_np
            if parting_interface_np is not None
            else np.zeros_like(hair_volume, dtype=bool)
        ),
    )
    print(f"  Fusion debug saved.")

    print(f"  Hair volume: {hair_volume.sum()} voxels "
          f"({100*hair_volume.sum()/hair_volume.size:.1f}%)")

    # ================================================================
    #  4. Build PDE strategy with fused boundary
    # ================================================================
    print(f"\nStep 4: Building orientation field ({args.pde_solver_mode})...")

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
    opt_pde.pde_solver_mode = args.pde_solver_mode
    opt_pde.pde_side_soft_confidence = args.pde_side_soft_confidence
    opt_pde.pde_side_screening_length = args.pde_side_screening_length
    opt_pde.pde_side_max_angle_degrees = args.pde_side_max_angle_degrees
    opt_pde.pde_side_constraint_mode = args.pde_side_constraint_mode
    opt_pde.pde_domain_tangent_confidence = args.pde_domain_tangent_confidence
    opt_pde.pde_domain_tangent_length = args.pde_domain_tangent_length
    opt_pde.pde_domain_padding_voxels = args.pde_domain_padding_voxels
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
            internal_interface_mask=parting_interface_np,
            internal_interface_normals=parting_interface_normals_np,
            internal_interface_confidence=(
                args.front_parting_interface_confidence
            ),
            internal_interface_length=args.front_parting_interface_length,
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
    if getattr(strategy, "solver_metrics", None) is not None:
        metrics_path = os.path.join(out_dir, "pde_solver_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(
                strategy.solver_metrics,
                metrics_file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  PDE solver metrics saved: {metrics_path}")

    # Keep the solved orientation inside the narrow PDE domain. Expanding it
    # with nearest-neighbor EDT would introduce discontinuities into the field.
    np.save(
        os.path.join(out_dir, "debug_orien_vol.npy"),
        strategy._orien_vol,
    )
    print(f"  Orientation volume saved.")

    domain_sdf_vol = None
    domain_normal_vol = None
    domain_guard_margin = 0.0
    if args.rk4_domain_guard or args.rk4_local_domain_recovery:
        domain_sdf, domain_normals, domain_spacing = build_signed_domain_distance(
            strategy._occ_vol > 0.5,
            b_min_val,
            b_max_val,
        )
        domain_guard_margin = (
            float(args.rk4_domain_guard_margin)
            if args.rk4_domain_guard_margin > 0.0
            else 2.0 * float(np.max(domain_spacing))
        )
        domain_sdf_vol = torch.from_numpy(domain_sdf).unsqueeze(0).unsqueeze(0).to(cuda)
        domain_normal_vol = torch.from_numpy(domain_normals).unsqueeze(0).to(cuda)
        np.save(os.path.join(out_dir, "pde_domain_sdf.npy"), domain_sdf)
        print(
            "  RK4 PDE-domain guard: "
            f"margin={domain_guard_margin:.5f}m, "
            f"inward_bias={args.rk4_domain_inward_bias:.3f}, "
            f"projection_epsilon={args.rk4_domain_projection_epsilon:.5f}m"
        )
    local_domain_recovery_margin = (
        domain_guard_margin
        if args.rk4_local_domain_recovery_margin <= 0.0
        else float(args.rk4_local_domain_recovery_margin)
    )

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
    surface_attraction_vol = None
    if surface_attraction_np is not None:
        surface_attraction_vol = (
            torch.from_numpy(surface_attraction_np).unsqueeze(0).to(cuda)
        )
        del surface_attraction_np
    parting_interface_vol = None
    parting_interface_normal_vol = None
    if parting_interface_np is not None:
        parting_interface_vol = (
            torch.from_numpy(parting_interface_np.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(cuda)
        )
        parting_interface_normal_vol = (
            torch.from_numpy(parting_interface_normals_np)
            .unsqueeze(0)
            .to(cuda)
        )

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
    parting_reseed_mask = None
    if args.front_parting_reseed_roots:
        if front_parting_mask is None:
            raise ValueError(
                "--front_parting_reseed_roots 需要 --front_parting_mask"
            )
        reseeded_roots, reseeded_mask_np, reseed_report = (
            reseed_roots_along_parting_boundaries(
                root_tensor.squeeze(0).T.detach().cpu().numpy(),
                head_mesh,
                geometry_calibs["front"],
                front_parting_mask,
                band_radius_px=args.front_parting_reseed_radius_px,
                boundary_offset_px=args.front_parting_reseed_offset_px,
                surface_distance=args.front_parting_reseed_surface_distance,
            )
        )
        root_tensor = (
            torch.from_numpy(reseeded_roots.T).float().unsqueeze(0).to(cuda)
        )
        parting_reseed_mask = torch.from_numpy(reseeded_mask_np).to(cuda)
        with open(
            os.path.join(out_dir, "parting_root_reseed_report.json"),
            "w",
            encoding="utf-8",
        ) as reseed_file:
            json.dump(reseed_report, reseed_file, ensure_ascii=False, indent=2)
        print(f"  Parting boundary root reseed: {reseed_report}")
    if args.project_roots_to_head:
        roots_flat = root_tensor.squeeze(0)
        projected_roots, projected_count = project_roots_to_head_surface(
            roots_flat,
            sdf_vol,
            normal_vol,
            b_min_t,
            b_max_t,
            cuda,
            target_distance=args.root_surface_distance,
            iterations=args.root_projection_iterations,
        )
        root_tensor = projected_roots.reshape(1, 3, -1)
        print(
            f"[Root Projection] corrected samples={projected_count}, "
            f"target={args.root_surface_distance:.4f}m, "
            f"iterations={args.root_projection_iterations}"
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
    depth_bias_direction = None
    root_depth_bias = None
    root_depth_offset = None
    root_layers = None
    if (
        abs(args.front_depth_bias) > 0.0
        or args.root_layer_depth_bias > 0.0
        or args.root_layer_depth_offset > 0.0
    ):
        front_inverse = np.linalg.inv(
            np.asarray(geometry_calibs["front"][0], dtype=np.float64)
        )

        def unproject_front_center(depth):
            clip = np.array([0.0, 0.0, depth, 1.0], dtype=np.float64)
            world_h = front_inverse @ clip
            return world_h[:3] / world_h[3]

        front_ray = (
            unproject_front_center(-0.5) - unproject_front_center(1.5)
        )
        front_ray /= np.linalg.norm(front_ray) + 1e-12
        depth_bias_direction = torch.from_numpy(
            front_ray.astype(np.float32)
        ).to(cuda).unsqueeze(1)
        root_depth = torch.sum(
            root_tensor.squeeze(0) * depth_bias_direction, dim=0
        )
        depth_center = torch.quantile(root_depth, 0.50)
        depth_low = torch.quantile(root_depth, 0.05)
        depth_high = torch.quantile(root_depth, 0.95)
        delta = root_depth - depth_center
        negative_scale = torch.clamp(depth_center - depth_low, min=1e-6)
        positive_scale = torch.clamp(depth_high - depth_center, min=1e-6)
        root_layers = torch.where(
            delta < 0.0,
            delta / negative_scale,
            delta / positive_scale,
        ).clamp(-1.0, 1.0)
        root_depth_bias = (
            float(args.front_depth_bias)
            + float(args.root_layer_depth_bias) * root_layers
        )
        root_depth_offset = float(args.root_layer_depth_offset) * root_layers
        print(
            "  Front camera depth bias: "
            f"global={args.front_depth_bias:.3f}, "
            f"root_layer={args.root_layer_depth_bias:.3f}, "
            f"layer_offset={args.root_layer_depth_offset:.3f}m, "
            f"ray={front_ray}; "
            f"layer_q05/50/95={float(torch.quantile(root_layers, 0.05)):.2f}/"
            f"{float(torch.quantile(root_layers, 0.50)):.2f}/"
            f"{float(torch.quantile(root_layers, 0.95)):.2f}"
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

    roots_3d = root_tensor.squeeze(0).reshape(3, -1)
    N_roots = roots_3d.shape[1]
    front_visible_np = compute_root_head_visibility(
        "data/head_model.obj",
        roots_3d.T.detach().cpu().numpy(),
        geometry_calibs["front"][0],
    )
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
    parting_root_reference = None
    parting_barrier_center = None
    parting_barrier_side = None
    parting_barrier_y_bounds = None
    parting_barrier_image_shape = None
    parting_surface_follow_mask = (
        parting_reseed_mask.clone()
        if parting_reseed_mask is not None else None
    )
    if front_parting_mask is not None:
        root_scalp_normals = query_grid(
            normal_vol, roots_3d, b_min_t, b_max_t
        ).T.detach().cpu().numpy()
        reference_np, influence_np = build_front_parting_root_reference(
            roots_3d.T.detach().cpu().numpy(),
            geometry_calibs["front"],
            strand_maps["front"],
            front_parting_mask,
            radius_px=args.front_parting_root_sign_radius_px,
            scalp_normals=root_scalp_normals,
        )
        parting_root_reference = torch.from_numpy(reference_np.T).to(cuda)
        print(
            "  Parting root image-flow guidance: "
            f"guided={int(influence_np.sum())}/{N_roots}, "
            f"radius={args.front_parting_root_sign_radius_px:.1f}px"
        )
        if args.front_parting_scalp_geodesic_reference:
            reference_np, influence_np, scalp_curve_np = (
                build_scalp_parting_geodesic_reference(
                    roots_3d.T.detach().cpu().numpy(),
                    geometry_calibs["front"],
                    front_parting_mask,
                    root_scalp_normals,
                    head_mesh=head_mesh,
                    radius_px=args.front_parting_geodesic_radius_px,
                    endpoint_blend_px=(
                        args.front_parting_geodesic_endpoint_blend_px
                    ),
                    curve_neighbors=(
                        args.front_parting_geodesic_curve_neighbors
                    ),
                    curve_mode=args.front_parting_geodesic_curve_mode,
                )
            )
            if args.front_parting_geodesic_curve_mode == "mesh_raycast":
                influence_np &= front_visible_np
                reference_np[~influence_np] = 0.0
            parting_root_reference = torch.from_numpy(reference_np.T).to(cuda)
            geodesic_follow = torch.from_numpy(influence_np).to(cuda)
            parting_surface_follow_mask = (
                geodesic_follow
                if parting_surface_follow_mask is None
                else (parting_surface_follow_mask | geodesic_follow)
            )
            np.savez_compressed(
                os.path.join(out_dir, "front_parting_scalp_curve.npz"),
                points=scalp_curve_np,
                influenced_roots=influence_np,
                root_reference=reference_np,
            )
            print(
                "  Parting 3D scalp-geodesic guidance: "
                f"curve_points={len(scalp_curve_np)}, "
                f"guided={int(influence_np.sum())}/{N_roots}, "
                f"radius={args.front_parting_geodesic_radius_px:.1f}px, "
                f"curve_mode={args.front_parting_geodesic_curve_mode}, "
                "saved=front_parting_scalp_curve.npz"
            )
        if (
            args.front_parting_direction_barrier_radius_px > 0.0
            or args.front_parting_pde_interface
        ):
            center_np, side_np, y_bounds = build_front_parting_barrier_data(
                roots_3d.T.detach().cpu().numpy(),
                geometry_calibs["front"],
                front_parting_mask,
            )
            parting_barrier_center = torch.from_numpy(center_np).to(cuda)
            parting_barrier_side = torch.from_numpy(side_np).to(cuda)
            parting_barrier_y_bounds = y_bounds
            parting_barrier_image_shape = front_parting_mask.shape
            cap_follow_np = select_front_parting_back_cap_roots(
                root_tensor.squeeze(0).T.detach().cpu().numpy(),
                geometry_calibs["front"],
                center_np,
                y_bounds[0],
                args.front_parting_direction_barrier_back_cap_radius_px,
                front_parting_mask.shape,
            )
            if np.any(cap_follow_np):
                cap_follow = torch.from_numpy(cap_follow_np).to(cuda)
                parting_surface_follow_mask = (
                    cap_follow
                    if parting_surface_follow_mask is None
                    else (parting_surface_follow_mask | cap_follow)
                )
            print(
                "  Parting dynamic direction barrier: "
                f"roots={int(np.count_nonzero(side_np))}/{N_roots}, "
                f"radius={args.front_parting_direction_barrier_radius_px:.1f}px, "
                f"rows={y_bounds}, "
                f"back_cap_follow_roots={int(np.count_nonzero(cap_follow_np))}"
            )
    uv_px = uv_px_float.long().clamp(0, 511)
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
    use_layer_clustering = (
        root_layers is not None
        and args.root_layer_cluster_scale_px > 0.0
        and (
            args.root_layer_depth_bias > 0.0
            or args.root_layer_depth_offset > 0.0
        )
    )
    if use_layer_clustering:
        selectable_layers = root_layers[selectable_ids]
        cluster_features = np.column_stack([
            uv_np,
            (
                selectable_layers * args.root_layer_cluster_scale_px
            ).detach().cpu().numpy(),
        ])
        print(
            "  Depth-aware guide clustering: "
            f"scale={args.root_layer_cluster_scale_px:.1f}px"
        )
    else:
        cluster_features = uv_np
    num_guides = min(1024, len(selectable_ids))
    kmeans = MiniBatchKMeans(
        n_clusters=num_guides, random_state=42, n_init="auto", batch_size=2048
    ).fit(cluster_features)
    centroids_list = kmeans.cluster_centers_.tolist()

    guide_idx_list = []
    for centroid in centroids_list:
        cx, cy = centroid[:2]
        candidate_uv = uv_px_float[:, selectable_ids]
        dist_sq = (candidate_uv[0] - cx) ** 2 + (candidate_uv[1] - cy) ** 2
        if use_layer_clustering:
            depth_center = centroid[2]
            dist_sq = dist_sq + (
                selectable_layers * args.root_layer_cluster_scale_px
                - depth_center
            ) ** 2
        guide_idx_list.append(selectable_ids[torch.argmin(dist_sq)].item())

    guide_idx_tensor = torch.tensor(guide_idx_list, device=cuda)
    guide_roots = root_tensor[:, :, guide_idx_tensor]

    centroids_t = torch.tensor(centroids_list, device=cuda)

    # Nearest-centroid mapping for all roots
    dist_sq_all = (
        (uv_px_float[0].unsqueeze(0) - centroids_t[:, 0].unsqueeze(1)) ** 2
        + (uv_px_float[1].unsqueeze(0) - centroids_t[:, 1].unsqueeze(1)) ** 2
    )
    if use_layer_clustering:
        dist_sq_all = dist_sq_all + (
            root_layers.unsqueeze(0) * args.root_layer_cluster_scale_px
            - centroids_t[:, 2].unsqueeze(1)
        ) ** 2
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
        root_tangent_steps=args.root_tangent_steps,
        root_tangent_strength=args.root_tangent_strength,
        root_surface_follow_mask=(
            parting_surface_follow_mask[guide_idx_tensor]
            if parting_surface_follow_mask is not None else None
        ),
        root_surface_follow_steps=args.front_parting_surface_follow_steps,
        root_surface_follow_distance=(
            args.front_parting_surface_follow_distance
        ),
        surface_attraction_vol=surface_attraction_vol,
        surface_attraction_weight=args.surface_attraction_weight,
        surface_attraction_deadzone=args.surface_attraction_deadzone,
        surface_attraction_full_distance=args.surface_attraction_full_distance,
        depth_bias_direction=depth_bias_direction,
        depth_bias_weight=(
            root_depth_bias[guide_idx_tensor]
            if root_depth_bias is not None else 0.0
        ),
        depth_layer_offset=(
            root_depth_offset[guide_idx_tensor]
            if root_depth_offset is not None else None
        ),
        domain_sdf_vol=domain_sdf_vol,
        domain_normal_vol=domain_normal_vol,
        domain_guard_margin=domain_guard_margin,
        domain_guard_inward_bias=args.rk4_domain_inward_bias,
        domain_projection_epsilon=args.rk4_domain_projection_epsilon,
        domain_projection_enabled=args.rk4_domain_guard,
        local_domain_recovery=args.rk4_local_domain_recovery,
        local_domain_recovery_margin=local_domain_recovery_margin,
        local_domain_recovery_max_angle_degrees=(
            args.rk4_local_domain_recovery_max_angle_degrees
        ),
        root_direction_reference=(
            parting_root_reference[:, guide_idx_tensor]
            if parting_root_reference is not None else None
        ),
        root_direction_reference_steps=args.front_parting_root_sign_steps,
        root_direction_min_component=(
            args.front_parting_root_direction_min_component
        ),
        parting_barrier_center=parting_barrier_center,
        parting_barrier_side=(
            parting_barrier_side[guide_idx_tensor]
            if parting_barrier_side is not None else None
        ),
        parting_barrier_y_bounds=parting_barrier_y_bounds,
        parting_barrier_image_shape=parting_barrier_image_shape,
        parting_barrier_radius_px=(
            args.front_parting_direction_barrier_radius_px
        ),
        parting_barrier_min_component=(
            args.front_parting_direction_barrier_min_component
        ),
        parting_barrier_margin_px=(
            args.front_parting_direction_barrier_margin_px
        ),
        parting_barrier_back_taper_px=(
            args.front_parting_direction_barrier_back_taper_px
        ),
        parting_barrier_back_cap_radius_px=(
            args.front_parting_direction_barrier_back_cap_radius_px
        ),
        parting_interface_vol=parting_interface_vol,
        parting_interface_normal_vol=parting_interface_normal_vol,
        root_collision_ramp_steps=args.rk4_root_collision_ramp_steps,
        root_collision_start_distance=(
            args.rk4_root_collision_start_distance
        ),
        label="guides",
    )
    guide_strands_t = (
        torch.from_numpy(guide_strands).to(cuda).permute(1, 2, 0)
    )

    # Trace all strands with guide clustering
    print(f"  Tracing {N_roots} full strands...")
    valid_cluster_mask = selectable_roots.float()

    strands, rk4_diagnostics = hair_synthesis_rk4(
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
        root_tangent_steps=args.root_tangent_steps,
        root_tangent_strength=args.root_tangent_strength,
        root_surface_follow_mask=parting_surface_follow_mask,
        root_surface_follow_steps=args.front_parting_surface_follow_steps,
        root_surface_follow_distance=(
            args.front_parting_surface_follow_distance
        ),
        surface_attraction_vol=surface_attraction_vol,
        surface_attraction_weight=args.surface_attraction_weight,
        surface_attraction_deadzone=args.surface_attraction_deadzone,
        surface_attraction_full_distance=args.surface_attraction_full_distance,
        depth_bias_direction=depth_bias_direction,
        depth_bias_weight=(root_depth_bias if root_depth_bias is not None else 0.0),
        depth_layer_offset=root_depth_offset,
        domain_sdf_vol=domain_sdf_vol,
        domain_normal_vol=domain_normal_vol,
        domain_guard_margin=domain_guard_margin,
        domain_guard_inward_bias=args.rk4_domain_inward_bias,
        domain_projection_epsilon=args.rk4_domain_projection_epsilon,
        domain_projection_enabled=args.rk4_domain_guard,
        local_domain_recovery=args.rk4_local_domain_recovery,
        local_domain_recovery_margin=local_domain_recovery_margin,
        local_domain_recovery_max_angle_degrees=(
            args.rk4_local_domain_recovery_max_angle_degrees
        ),
        root_direction_reference=parting_root_reference,
        root_direction_reference_steps=args.front_parting_root_sign_steps,
        root_direction_min_component=(
            args.front_parting_root_direction_min_component
        ),
        parting_barrier_center=parting_barrier_center,
        parting_barrier_side=parting_barrier_side,
        parting_barrier_y_bounds=parting_barrier_y_bounds,
        parting_barrier_image_shape=parting_barrier_image_shape,
        parting_barrier_radius_px=(
            args.front_parting_direction_barrier_radius_px
        ),
        parting_barrier_min_component=(
            args.front_parting_direction_barrier_min_component
        ),
        parting_barrier_margin_px=(
            args.front_parting_direction_barrier_margin_px
        ),
        parting_barrier_back_taper_px=(
            args.front_parting_direction_barrier_back_taper_px
        ),
        parting_barrier_back_cap_radius_px=(
            args.front_parting_direction_barrier_back_cap_radius_px
        ),
        parting_interface_vol=parting_interface_vol,
        parting_interface_normal_vol=parting_interface_normal_vol,
        root_collision_ramp_steps=args.rk4_root_collision_ramp_steps,
        root_collision_start_distance=(
            args.rk4_root_collision_start_distance
        ),
        return_diagnostics=True,
        label="strands",
    )
    diagnostics_path = os.path.join(out_dir, "rk4_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as diagnostics_file:
        json.dump(
            rk4_diagnostics,
            diagnostics_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  RK4 diagnostics saved: {diagnostics_path}")

    # ================================================================
    #  7. Save
    # ================================================================
    print(f"\nStep 7: Clipping & saving to {out_ply}...")
    if args.root_attach_steps > 0:
        strands, attached_points = attach_strand_roots_to_head(
            strands,
            sdf_vol,
            normal_vol,
            b_min_t,
            b_max_t,
            cuda,
            attach_steps=args.root_attach_steps,
            target_distance=args.root_attach_distance,
            strength=args.root_attach_strength,
        )
        print(
            f"[Root Attachment] corrected points={attached_points}, "
            f"steps={args.root_attach_steps}, "
            f"target={args.root_attach_distance:.4f}m"
        )
        strands, mesh_attached = project_strand_roots_to_mesh(
            strands,
            "data/head_model.obj",
            attach_steps=args.root_attach_steps,
            target_distance=args.root_attach_distance * 0.5,
        )
        print(
            f"[Mesh Root Attachment] corrected roots={mesh_attached}, "
            f"target={args.root_attach_distance * 0.5:.4f}m"
        )
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
    if front_parting_mask is not None and args.front_parting_trim_points > 0:
        strands, parting_trimmed = trim_strands_crossing_parting(
            strands,
            front_parting_mask,
            geometry_calibs["front"],
            trim_points=args.front_parting_trim_points,
            drop_reversed_tips=args.front_parting_drop_reversed_tips,
            head_mesh_path="data/head_model.obj",
            strand_root_visibility=front_visible_np,
            back_trim_px=args.front_parting_back_trim_px,
            drop_floating_arches=args.front_parting_drop_floating_arches,
            floating_arch_clearance=args.front_parting_floating_arch_clearance,
            floating_arch_steps=args.front_parting_floating_arch_steps,
        )
        print(
            f"[Front Parting Trim]回缩 {parting_trimmed} 条进入发缝禁区的末端，"
            f"points={args.front_parting_trim_points}"
        )
    if (
        front_parting_mask is not None
        and args.front_parting_surface_attach_steps > 0
    ):
        strands, surface_attached, hard_attached = (
            attach_parting_root_segments_to_mesh(
                strands,
                front_parting_mask,
                geometry_calibs["front"],
                "data/head_model.obj",
                attach_steps=args.front_parting_surface_attach_steps,
                hard_steps=args.front_parting_surface_hard_steps,
                target_distance=args.front_parting_surface_distance,
                influence_radius_px=args.front_parting_surface_radius_px,
                back_trim_px=args.front_parting_back_trim_px,
            )
        )
        print(
            "[Front Parting Surface Attachment] "
            f"strands={surface_attached}, hard_points={hard_attached}, "
            f"steps={args.front_parting_surface_attach_steps}, "
            f"hard_steps={args.front_parting_surface_hard_steps}, "
            f"target={args.front_parting_surface_distance:.4f}m"
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
