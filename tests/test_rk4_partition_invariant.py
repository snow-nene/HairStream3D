"""RK4 分区不变量回归测试。"""

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.recon_3d.run_pde_multiview import (
    enforce_partition_candidate,
    hair_synthesis_rk4,
    query_partition_labels,
    query_partitioned_grid,
    select_partitioned_guide_indices,
)


class _ConstantStrategy:
    def __init__(self, field, direction=(0.0, 1.0, 0.0)):
        self._orien_vol = np.asarray(field, dtype=np.float32)
        self.direction = torch.tensor(direction, dtype=torch.float32)
        self.query_calls = 0

    def query(self, points, calib):
        self.query_calls += 1
        batch, _, count = points.shape
        return self.direction.to(points.device).reshape(1, 3, 1).expand(
            batch, 3, count
        )

    def query_occ(self, points, calib):
        return torch.ones(
            (points.shape[0], 1, points.shape[2]), device=points.device
        )


def _partition_fixture():
    labels = torch.ones((1, 1, 8, 3, 3), dtype=torch.long)
    labels[:, :, 4:] = 2
    field = np.zeros((3, 8, 3, 3), dtype=np.float32)
    field[0, :4] = 1.0
    field[0, 4:] = -1.0
    b_min = torch.zeros(3, 1)
    b_max = torch.ones(3, 1)
    return labels, field, b_min, b_max


class RK4PartitionInvariantTests(unittest.TestCase):
    def test_partitioned_interpolation_excludes_other_label(self):
        labels, field, b_min, b_max = _partition_fixture()
        points = torch.tensor(
            [[0.49, 0.51], [0.5, 0.5], [0.5, 0.5]],
            dtype=torch.float32,
        )
        root_labels = torch.tensor([1, 2])

        sampled_labels = query_partition_labels(
            labels, points, b_min, b_max
        )
        values, support = query_partitioned_grid(
            torch.from_numpy(field).unsqueeze(0),
            labels,
            points,
            root_labels,
            b_min,
            b_max,
        )

        self.assertEqual(sampled_labels.tolist(), [1, 2])
        torch.testing.assert_close(
            values[0], torch.tensor([1.0, -1.0])
        )
        self.assertTrue(torch.all(support > 0.0))

    def test_cross_partition_candidate_is_rolled_back(self):
        labels, _, b_min, b_max = _partition_fixture()
        origin = torch.tensor([[0.49], [0.5], [0.5]])
        candidate = torch.tensor([[0.80], [0.5], [0.5]])

        corrected, crossed = enforce_partition_candidate(
            candidate,
            origin,
            torch.tensor([1]),
            labels,
            b_min,
            b_max,
        )

        self.assertTrue(bool(crossed.item()))
        self.assertGreaterEqual(float(corrected[0, 0]), 0.49)
        self.assertLess(float(corrected[0, 0]), 0.51)
        self.assertEqual(
            query_partition_labels(
                labels, corrected, b_min, b_max
            ).item(),
            1,
        )

    def test_invalid_root_partition_is_rejected(self):
        labels, field, b_min, b_max = _partition_fixture()
        strategy = _ConstantStrategy(field)
        root = torch.tensor([[[0.25], [0.5], [0.5]]])

        with self.assertRaisesRegex(ValueError, "positive root labels"):
            hair_synthesis_rk4(
                strategy,
                torch.device("cpu"),
                root,
                torch.eye(4).unsqueeze(0),
                num_sample=2,
                b_min_t=b_min,
                b_max_t=b_max,
                partition_label_vol=labels,
                root_partition_labels=torch.tensor([0]),
            )

    def test_guide_matching_never_crosses_partition(self):
        distances = torch.tensor(
            [
                [5.0, 1.0, 4.0, 0.1],
                [0.1, 4.0, 0.2, 3.0],
            ]
        )
        guide_labels = torch.tensor([1, 2])
        root_labels = torch.tensor([1, 2, 1, 2])

        indices = select_partitioned_guide_indices(
            distances, guide_labels, root_labels
        )

        self.assertEqual(indices.tolist(), [0, 1, 0, 1])
        self.assertTrue(torch.equal(guide_labels[indices], root_labels))

    def test_default_rk4_path_remains_partition_free(self):
        _, field, b_min, b_max = _partition_fixture()
        strategy = _ConstantStrategy(field)
        root = torch.tensor([[[0.25], [0.25], [0.25]]])

        strands, diagnostics = hair_synthesis_rk4(
            strategy,
            torch.device("cpu"),
            root,
            torch.eye(4).unsqueeze(0),
            num_sample=3,
            hair_unit=0.1,
            b_min_t=b_min,
            b_max_t=b_max,
            return_diagnostics=True,
            label="default-regression",
        )

        np.testing.assert_allclose(
            strands[0, :, 1], [0.25, 0.35, 0.45], atol=1e-6
        )
        self.assertGreater(strategy.query_calls, 0)
        self.assertFalse(diagnostics["partition_guard_enabled"])
        self.assertEqual(diagnostics["partition_crossing_candidates"], 0)
        self.assertEqual(diagnostics["final_partition_violation_strands"], 0)


if __name__ == "__main__":
    unittest.main()
