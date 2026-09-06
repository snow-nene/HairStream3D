import numpy as np
import pytest

from lib.surface_energy import build_surface_normal_penalty


def test_surface_penalty_is_symmetric_and_compact():
    distance = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=np.float32)
    normals = np.zeros((3, 1, 1, 4), dtype=np.float32)
    normals[2] = 1.0
    unit, weight = build_surface_normal_penalty(distance, normals, support_radius=1.0, max_weight=4.0)
    assert np.allclose(weight, [[[4.0, 2.0, 0.0, 0.0]]])
    assert np.allclose(unit[2], 1.0)


def test_internal_surface_and_degenerate_normal_are_rejected():
    distance = np.zeros((1, 1, 2), dtype=np.float32)
    normals = np.zeros((3, 1, 1, 2), dtype=np.float32)
    normals[0, 0, 0, 0] = 1.0
    labels = np.array([[[2, 1]]], dtype=np.int8)
    unit, weight = build_surface_normal_penalty(distance, normals, support_radius=1.0, max_weight=2.0, surface_type=labels)
    assert weight[0, 0, 0] == 0.0 and np.all(unit[:, 0, 0, 0] == 0.0)
    assert weight[0, 0, 1] == 0.0


def test_invalid_surface_energy_parameters():
    with pytest.raises(ValueError):
        build_surface_normal_penalty(np.zeros((1, 1, 1)), np.zeros((3, 1, 1, 1)), support_radius=0, max_weight=1)
