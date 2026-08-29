import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.recon_3d.attach_nearest_roots import (
    attach_nearest_roots,
    enforce_root_clearance_envelope,
    enforce_parting_safe_roots,
    filter_floating_parting_segments,
)


def test_parting_safe_root_moves_only_masked_root_to_local_scalp():
    mesh = trimesh.Trimesh(
        vertices=np.array([
            [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
        ]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    masked = np.column_stack([
        np.zeros(6), np.zeros(6), 0.01 + np.arange(6) * 0.01,
    ]).astype(np.float32)
    safe = masked.copy()
    safe[:, 0] = -0.8
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 30:34] = True

    adjusted, report = enforce_parting_safe_roots(
        [masked, safe],
        mesh,
        mask,
        np.eye(4),
        boundary_offset_px=2,
        transition_steps=3,
        surface_distance=0.001,
        max_shift_distance=1.0,
    )
    root_pixels = (adjusted[0][0, 0] + 1.0) * 0.5 * 63

    assert report["violating_roots_before"] == 1
    assert report["corrected_roots"] == 1
    assert report["remaining_roots_in_parting"] == 0
    assert root_pixels < 30 or root_pixels > 33
    assert np.isclose(adjusted[0][0, 2], 0.001, atol=1e-4)
    assert np.allclose(adjusted[1], safe)


def test_automatically_reverses_and_attaches_nearest_endpoint():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    root = np.array([0.0, 0.0, 1.08], dtype=np.float32)
    outward = root + np.arange(10)[:, None] * np.array([0.0, 0.0, 0.05])
    reversed_input = outward[::-1].copy()

    attached, report = attach_nearest_roots(
        [outward, reversed_input],
        mesh,
        attach_steps=7,
        hard_steps=3,
        probe_steps=3,
        target_distance=0.0,
        endpoint_mode="nearest",
    )
    hard_points = np.concatenate([strand[:3] for strand in attached])
    _, distances, _ = trimesh.proximity.closest_point(mesh, hard_points)

    assert report["attached_strands"] == 2
    assert report["reversed_strands"] == 1
    assert distances.max() < 1e-5
    assert attached[0][-1, 2] > attached[0][0, 2]
    assert attached[1][-1, 2] > attached[1][0, 2]


def test_clearance_envelope_caps_early_lift_without_flattening_tip():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    strand = np.column_stack([
        np.zeros(12),
        np.zeros(12),
        1.0 + np.arange(12) * 0.006,
    ]).astype(np.float32)
    original_tip = strand[-1].copy()

    adjusted, report = enforce_root_clearance_envelope(
        [strand],
        mesh,
        contact_length=0.006,
        release_length=0.030,
        contact_clearance=0.0002,
        release_clearance=0.004,
    )
    _, distances, _ = trimesh.proximity.closest_point(mesh, adjusted[0][:6])

    assert report["adjusted_points"] > 0
    assert distances[0] <= 0.00021
    assert distances[1] <= 0.00021
    assert distances[3] < 0.003
    assert np.allclose(adjusted[0][-1], original_tip)


def test_parting_filter_rejects_only_floating_middle_or_tip_segments():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 29:35] = 1
    calib = np.eye(4, dtype=np.float64)
    root_z = np.sqrt(1.0 - 0.5 ** 2)
    floating = np.column_stack([
        np.linspace(-0.5, 0.0, 12),
        np.zeros(12),
        np.linspace(root_z, 1.15, 12),
    ]).astype(np.float32)
    valid = np.column_stack([
        np.full(12, -0.5),
        np.zeros(12),
        np.linspace(root_z, 1.10, 12),
    ]).astype(np.float32)

    filtered, report = filter_floating_parting_segments(
        [floating, valid],
        mesh,
        mask,
        calib,
        corridor_dilation_px=0,
        clearance=0.003,
        protected_root_steps=7,
        head_space="reconstruction",
    )

    assert report["rejected_strands"] == 1
    assert len(filtered) == 1
    assert np.allclose(filtered[0], valid)


if __name__ == "__main__":
    test_parting_safe_root_moves_only_masked_root_to_local_scalp()
    test_automatically_reverses_and_attaches_nearest_endpoint()
    test_clearance_envelope_caps_early_lift_without_flattening_tip()
    test_parting_filter_rejects_only_floating_middle_or_tip_segments()
    print("test_nearest_root_postprocess: 4 passed")
