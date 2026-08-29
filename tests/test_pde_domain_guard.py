import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.recon_strategy.weighted_poisson import (
    build_signed_domain_distance,
    limit_outward_domain_direction,
    recover_points_inside_domain,
)


class PdeDomainGuardTests(unittest.TestCase):
    def test_signed_distance_uses_metric_spacing_and_inward_normals(self):
        domain = np.zeros((9, 7, 5), dtype=bool)
        domain[2:7, 1:6, 1:4] = True
        signed, normals, spacing = build_signed_domain_distance(
            domain,
            b_min=(-0.4, 0.0, -0.2),
            b_max=(0.4, 1.2, 0.2),
        )

        np.testing.assert_allclose(spacing, (0.1, 0.2, 0.1))
        self.assertGreater(float(signed[4, 3, 2]), 0.0)
        self.assertLess(float(signed[1, 3, 2]), 0.0)
        self.assertGreater(float(normals[0, 1, 3, 2]), 0.9)
        self.assertLess(float(normals[0, 7, 3, 2]), -0.9)

    def test_outward_component_turns_inward_without_losing_speed(self):
        direction = torch.tensor(
            [[-1.0, -1.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]
        )
        normal = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        signed = torch.tensor([0.001, 0.02, 0.001])

        limited = limit_outward_domain_direction(
            direction, signed, normal, margin=0.005, inward_bias=0.05
        )

        self.assertGreater(float(limited[0, 0]), 0.0)
        torch.testing.assert_close(
            torch.linalg.norm(limited[:, 0]),
            torch.linalg.norm(direction[:, 0]),
        )
        np.testing.assert_allclose(limited[:, 1].numpy(), direction[:, 1].numpy())
        np.testing.assert_allclose(limited[:, 2].numpy(), direction[:, 2].numpy())

    def test_local_recovery_projects_only_near_boundary_points(self):
        points = torch.zeros(3, 3)
        signed = torch.tensor([-0.01, 0.001, 0.02])
        normal = torch.zeros(3, 3)
        normal[0] = 1.0
        recovered = recover_points_inside_domain(
            points, signed, normal, margin=0.005, epsilon=0.001
        )

        torch.testing.assert_close(
            recovered[:, 0], torch.tensor([0.016, 0.0, 0.0])
        )
        torch.testing.assert_close(
            recovered[:, 1], torch.tensor([0.005, 0.0, 0.0])
        )
        torch.testing.assert_close(recovered[:, 2], points[:, 2])


if __name__ == "__main__":
    unittest.main()
