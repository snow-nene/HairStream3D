import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.recon_3d.run_pde_multiview import (
    apply_parting_direction_barrier,
    attach_parting_root_segments_to_mesh,
    build_front_parting_corridor,
    build_front_parting_interface_constraints,
    build_front_parting_root_reference,
    build_scalp_parting_geodesic_reference,
    enforce_parting_interface_candidate,
    parting_back_taper_weight,
    project_parting_candidate_to_side,
    reseed_roots_along_parting_boundaries,
    root_collision_distance_for_step,
    select_front_parting_back_cap_roots,
    taper_front_parting_domain_mask,
)


def test_parting_back_taper_opens_smoothly_from_rear_endpoint():
    y = torch.tensor([20.0, 23.0, 26.0, 29.0, 32.0, 40.0])
    weight = parting_back_taper_weight(y, active_y_min=20, taper_px=12)

    assert torch.isclose(weight[0], torch.tensor(0.0))
    assert torch.isclose(weight[2], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(weight[4], torch.tensor(1.0))
    assert torch.isclose(weight[5], torch.tensor(1.0))
    assert torch.all(weight[1:] >= weight[:-1])


def test_parting_domain_mask_narrows_smoothly_at_rear_endpoint():
    mask = np.zeros((24, 20), dtype=bool)
    mask[4:20, 7:13] = True

    tapered = taper_front_parting_domain_mask(mask, back_taper_px=8)

    assert not tapered[4].any()
    assert 0 < tapered[8].sum() < mask[8].sum()
    assert np.array_equal(tapered[12], mask[12])
    assert np.array_equal(tapered[19], mask[19])
    assert not tapered[:4].any()


def test_parting_corridor_can_be_restricted_to_a_thin_scalp_layer():
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 30:34] = True
    domain = np.ones((32, 32, 32), dtype=bool)
    mesh = trimesh.Trimesh(
        vertices=np.array([
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )

    full = build_front_parting_corridor(
        mask, np.eye(4), [-1, -1, -1], [1, 1, 1], 32, domain,
        image_dilation_px=0, voxel_dilation=0, top_extension_px=0,
    )
    thin = build_front_parting_corridor(
        mask, np.eye(4), [-1, -1, -1], [1, 1, 1], 32, domain,
        image_dilation_px=0, voxel_dilation=0, top_extension_px=0,
        surface_mesh=mesh, surface_max_distance=0.2,
    )

    z = np.linspace(-1.0, 1.0, 32)[None, None, :]
    assert thin.any()
    assert thin.sum() < full.sum()
    assert np.all(np.abs(np.broadcast_to(z, thin.shape)[thin]) <= 0.2)


def test_parting_interface_normal_is_lateral_and_scalp_tangent():
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 30:34] = True
    domain = np.ones((32, 32, 32), dtype=bool)
    mesh = trimesh.Trimesh(
        vertices=np.array([
            [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
        ]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )

    interface, normals = build_front_parting_interface_constraints(
        mask, np.eye(4), [-1, -1, -1], [1, 1, 1], 32, domain,
        mesh, surface_max_distance=0.2,
    )

    values = normals[:, interface]
    assert interface.any()
    assert np.all(values[0] > 0.99)
    assert np.allclose(values[1:], 0.0, atol=1e-6)


def test_rk4_parting_interface_returns_both_sides_to_their_root_side():
    mask = torch.zeros(1, 1, 5, 5, 5)
    mask[0, 0, 2, 2, 2] = 1.0
    normals = torch.zeros(1, 3, 5, 5, 5)
    normals[0, 0, 2, 2, 2] = 1.0
    candidate = torch.zeros(3, 2)
    origin = torch.tensor([[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
    bounds_min = torch.full((3, 1), -1.0)
    bounds_max = torch.full((3, 1), 1.0)

    corrected, hit, frozen = enforce_parting_interface_candidate(
        candidate,
        origin,
        torch.tensor([-1.0, 1.0]),
        mask,
        normals,
        bounds_min,
        bounds_max,
    )

    assert hit.tolist() == [True, True]
    assert corrected[0, 0] < 0.0
    assert corrected[0, 1] > 0.0
    assert not frozen.any()


def test_root_collision_shell_grows_without_first_step_jump():
    distances = [
        root_collision_distance_for_step(
            step, ramp_steps=6, start_distance=0.0015
        )
        for step in range(1, 8)
    ]

    assert np.isclose(distances[0], 0.0015)
    assert np.isclose(distances[5], 0.005)
    assert np.isclose(distances[6], 0.005)
    assert np.all(np.diff(distances) >= 0.0)
    assert root_collision_distance_for_step(1, ramp_steps=0) == 0.005


def test_parting_candidate_projection_uses_left_and_right_mask_edges():
    points = torch.tensor([
        [0.0, 0.0, -0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    boundaries = torch.stack([
        torch.full((64,), 30.0),
        torch.full((64,), 33.0),
    ])
    side = torch.tensor([-1.0, 1.0, -1.0, 1.0])

    corrected, projected = project_parting_candidate_to_side(
        points,
        torch.eye(4).unsqueeze(0),
        boundaries,
        side,
        image_height=64,
        image_width=64,
        active_y_min=20,
        active_y_max=44,
        margin_px=1.0,
    )

    corrected_px = (corrected[0] + 1.0) * 0.5 * 63
    assert projected.tolist() == [True, True, False, False]
    assert torch.isclose(corrected_px[0], torch.tensor(29.0), atol=1e-5)
    assert torch.isclose(corrected_px[1], torch.tensor(34.0), atol=1e-5)
    assert torch.allclose(corrected[:, 2:], points[:, 2:])


def test_parting_direction_barrier_only_corrects_inward_velocity():
    points = torch.tensor([
        [-0.1, 0.1, -0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    direction = torch.tensor([
        [1.0, -1.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    center = torch.full((64,), 31.5)
    side = torch.tensor([-1.0, 1.0, -1.0, 1.0])

    corrected, active, cap_active = apply_parting_direction_barrier(
        direction,
        points,
        torch.eye(4).unsqueeze(0),
        center,
        side,
        image_height=64,
        image_width=64,
        active_y_min=20,
        active_y_max=44,
        radius_px=10.0,
        min_outward_component=0.1,
    )

    assert active.tolist() == [True, True, False, False]
    assert not cap_active.any()
    assert corrected[0, 0] < 0.0
    assert corrected[0, 1] > 0.0
    assert torch.allclose(corrected[:, 2:], direction[:, 2:])


def test_parting_rear_cap_turns_horizontal_flow_toward_posterior():
    image_y = 15.0
    ndc_y = image_y / 63.0 * 2.0 - 1.0
    points = torch.tensor([[0.0], [ndc_y], [0.0]])
    direction = torch.tensor([[1.0], [0.0], [0.0]])
    boundaries = torch.stack([
        torch.full((64,), 30.0),
        torch.full((64,), 33.0),
    ])

    corrected, active, cap_active = apply_parting_direction_barrier(
        direction,
        points,
        torch.eye(4).unsqueeze(0),
        boundaries,
        torch.tensor([-1.0]),
        image_height=64,
        image_width=64,
        active_y_min=20,
        active_y_max=44,
        radius_px=10.0,
        min_outward_component=0.2,
        back_taper_px=12.0,
        back_cap_radius_px=12.0,
    )

    assert active.item()
    assert cap_active.item()
    assert corrected[1, 0] < 0.0


def test_parting_rear_cap_roots_are_selected_for_surface_following():
    ndc_y_near = 15.0 / 63.0 * 2.0 - 1.0
    ndc_y_far = 2.0 / 63.0 * 2.0 - 1.0
    roots = np.array([
        [0.0, ndc_y_near, 0.0],
        [0.0, ndc_y_far, 0.0],
        [0.8, ndc_y_near, 0.0],
    ])
    boundaries = np.stack([
        np.full(64, 30.0),
        np.full(64, 33.0),
    ])

    selected = select_front_parting_back_cap_roots(
        roots,
        np.eye(4),
        boundaries,
        active_y_min=20,
        radius_px=12,
        image_shape=(64, 64),
    )

    assert selected.tolist() == [True, False, False]


def test_only_parting_near_root_segment_is_attached_to_surface():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    strands = np.zeros((2, 8, 3), dtype=np.float32)
    near_root = np.array([0.04, 0.0, np.sqrt(1.0 - 0.04**2) + 0.01])
    far_root = np.array([0.75, 0.0, np.sqrt(1.0 - 0.75**2) + 0.01])
    strands[0] = near_root + np.arange(8)[:, None] * np.array([0.0, 0.0, 0.05])
    strands[1] = far_root + np.arange(8)[:, None] * np.array([0.0, 0.0, 0.05])
    original_far = strands[1].copy()
    original_near_tip = strands[0, -1].copy()

    mask = np.zeros((64, 64), dtype=bool)
    mask[24:41, 31:34] = True
    calib = np.eye(4, dtype=np.float32)
    attached, strand_count, hard_point_count = (
        attach_parting_root_segments_to_mesh(
            strands,
            mask,
            calib,
            mesh,
            attach_steps=8,
            hard_steps=3,
            target_distance=0.02,
            influence_radius_px=5.0,
        )
    )

    _, distances, _ = trimesh.proximity.closest_point(mesh, attached[0, :3])
    assert strand_count == 1
    assert hard_point_count == 3
    assert np.allclose(distances, 0.02, atol=2e-3)
    assert np.allclose(attached[1], original_far)
    assert np.allclose(attached[0, -1], original_near_tip)


def test_parting_reference_keeps_image_flow_and_only_flips_its_sign():
    roots = np.array([
        [-0.2, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    strand_map = np.zeros((64, 64, 3), dtype=np.float32)
    strand_map[:, :, 0] = 1.0
    strand_map[:, :, 1] = 0.75  # positive image-y component
    strand_map[:, :, 2] = 0.0   # positive image-x component
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 31:34] = True
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))

    reference, influence = build_front_parting_root_reference(
        roots,
        np.eye(4, dtype=np.float32),
        strand_map,
        mask,
        radius_px=20,
        scalp_normals=normals,
    )

    assert influence.tolist() == [True, True]
    assert reference[0, 0] < 0.0
    assert reference[1, 0] > 0.0
    assert abs(reference[0, 1]) > 0.1
    assert abs(reference[1, 1]) > 0.1
    assert np.allclose(reference[:, 2], 0.0, atol=1e-6)


def test_scalp_geodesic_reference_separates_sides_and_turns_at_rear_endpoint():
    rows = np.array([20.0, 28.0, 36.0, 44.0])
    roots_px = np.array(
        [[x, y] for y in rows for x in (25.0, 39.0)], dtype=np.float64
    )
    roots = np.column_stack([
        roots_px[:, 0] / 63.0 * 2.0 - 1.0,
        roots_px[:, 1] / 63.0 * 2.0 - 1.0,
        np.zeros(len(roots_px)),
    ])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(roots), 1))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 31:33] = True

    reference, influence, curve = build_scalp_parting_geodesic_reference(
        roots,
        np.eye(4, dtype=np.float32),
        mask,
        normals,
        radius_px=20.0,
        endpoint_blend_px=12.0,
        curve_neighbors=2,
    )

    assert influence.all()
    assert len(curve) == 25
    # Away from the rear endpoint, roots point outwards on each side.
    assert reference[4, 0] < 0.0
    assert reference[5, 0] > 0.0
    # At the image-top/rear endpoint both sides turn toward decreasing y.
    assert reference[0, 1] < 0.0
    assert reference[1, 1] < 0.0
    assert np.allclose(reference[:, 2], 0.0, atol=1e-6)


def test_scalp_geodesic_raycast_uses_visible_mesh_intersection():
    rows = np.array([20.0, 28.0, 36.0, 44.0])
    roots_px = np.array(
        [[x, y] for y in rows for x in (25.0, 39.0)], dtype=np.float64
    )
    roots = np.column_stack([
        roots_px[:, 0] / 63.0 * 2.0 - 1.0,
        roots_px[:, 1] / 63.0 * 2.0 - 1.0,
        np.full(len(roots_px), 0.4),
    ])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(roots), 1))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 31:33] = True
    mesh = trimesh.Trimesh(
        vertices=np.array([
            [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
        ]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )

    reference, influence, curve = build_scalp_parting_geodesic_reference(
        roots,
        np.eye(4, dtype=np.float32),
        mask,
        normals,
        head_mesh=mesh,
        radius_px=20.0,
        endpoint_blend_px=12.0,
        curve_mode="mesh_raycast",
    )

    assert influence.all()
    assert np.allclose(curve[:, 2], 0.0, atol=1e-6)
    assert np.max(np.abs(curve[:, 0])) < 0.02
    assert reference[0, 1] < 0.0
    assert reference[1, 1] < 0.0


def test_parting_reseed_moves_roots_to_both_mask_edges_on_scalp():
    vertices = np.array([
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    roots = np.array([
        [-0.05, 0.0, 0.4],
        [0.05, 0.0, 0.4],
    ], dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 31:34] = True

    reseeded, selected, report = reseed_roots_along_parting_boundaries(
        roots,
        mesh,
        np.eye(4, dtype=np.float32),
        mask,
        band_radius_px=4,
        boundary_offset_px=2,
        surface_distance=0.001,
        max_shift_distance=1.0,
    )

    assert selected.tolist() == [True, True]
    assert report["left_roots"] == 1
    assert report["right_roots"] == 1
    assert reseeded[0, 0] < roots[0, 0]
    assert reseeded[1, 0] > roots[1, 0]
    assert np.allclose(reseeded[:, 2], 0.001, atol=2e-4)


if __name__ == "__main__":
    test_parting_back_taper_opens_smoothly_from_rear_endpoint()
    test_parting_candidate_projection_uses_left_and_right_mask_edges()
    test_parting_direction_barrier_only_corrects_inward_velocity()
    test_only_parting_near_root_segment_is_attached_to_surface()
    test_parting_reference_keeps_image_flow_and_only_flips_its_sign()
    test_parting_reseed_moves_roots_to_both_mask_edges_on_scalp()
    print("test_parting_root_attachment: 6 passed")
