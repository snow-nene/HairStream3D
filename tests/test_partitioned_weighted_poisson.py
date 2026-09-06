"""分区 screened-Poisson 的隔离回归测试。"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson
from scripts.recon_3d.run_pde_multiview import build_front_partition_volume


def test_front_partition_labels_are_lifted_through_the_volume():
    labels = np.ones((9, 9), dtype=np.int16)
    labels[:, 4:] = 2
    labels[:, 4] = 0
    volume = build_front_partition_volume(
        labels,
        np.eye(4),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        resolution=7,
        chunk_size=2,
    )

    assert volume.shape == (7, 7, 7)
    assert np.all(volume > 0)
    assert np.all(volume[:3] == 1)
    assert np.all(volume[4:] == 2)


def test_partition_labels_remove_cross_partition_diffusion():
    shape = (10, 3, 2)
    domain = np.ones(shape, dtype=bool)
    partitions = np.ones(shape, dtype=np.int16)
    partitions[5:] = 2
    boundary = np.zeros(shape, dtype=bool)
    boundary[0] = True
    boundary[-1] = True
    values = np.zeros((3, *shape), dtype=np.float64)
    values[0, 0] = -1.0
    values[0, -1] = 1.0

    field, metrics = solve_weighted_screened_poisson(
        domain,
        values,
        boundary,
        partition_labels=partitions,
        device="cpu",
        dtype=torch.float64,
        tolerance=1e-10,
        max_iterations=1000,
    )

    assert metrics.converged
    np.testing.assert_allclose(field[0, :5], -1.0, atol=1e-9)
    np.testing.assert_allclose(field[0, 5:], 1.0, atol=1e-9)


def test_partition_without_anchor_is_rejected():
    shape = (8, 3, 2)
    domain = np.ones(shape, dtype=bool)
    partitions = np.ones(shape, dtype=np.int16)
    partitions[4:] = 2
    boundary = np.zeros(shape, dtype=bool)
    boundary[0] = True
    values = np.zeros((3, *shape), dtype=np.float64)

    try:
        solve_weighted_screened_poisson(
            domain,
            values,
            boundary,
            partition_labels=partitions,
            device="cpu",
            dtype=torch.float64,
        )
    except ValueError as error:
        assert "unanchored connected components" in str(error)
    else:
        raise AssertionError("没有锚点的 PDE 分区必须被拒绝")


if __name__ == "__main__":
    test_front_partition_labels_are_lifted_through_the_volume()
    test_partition_labels_remove_cross_partition_diffusion()
    test_partition_without_anchor_is_rejected()
    print("passed 3 partitioned weighted-Poisson tests")
