import numpy as np
import os
import sys
import torch
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.recon_strategy.weighted_poisson import (
    solve_weighted_screened_poisson,
)


def _solve(**kwargs):
    return solve_weighted_screened_poisson(
        device="cpu",
        dtype=torch.float64,
        tolerance=1e-8,
        max_iterations=1000,
        **kwargs,
    )


class WeightedPoissonTests(unittest.TestCase):
    def test_constant_dirichlet_field_is_preserved(self):
        shape = (9, 8, 7)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        value = np.array([0.25, -0.5, 0.75], dtype=np.float64)
        values = np.zeros((3, *shape), dtype=np.float64)
        values[:, boundary] = value[:, None]

        field, metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
        )

        np.testing.assert_allclose(
            field[:, domain],
            np.broadcast_to(value[:, None], field[:, domain].shape),
            atol=1e-8,
            rtol=0.0,
        )
        self.assertTrue(metrics.converged)
        self.assertLessEqual(metrics.final_relative_residual, 1e-8)
        self.assertEqual(metrics.dirichlet_max_error, 0.0)

    def test_linear_harmonic_field_matches_analytic_solution(self):
        shape = (13, 7, 5)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        boundary[-1] = True
        values = np.zeros((3, *shape), dtype=np.float64)
        values[0, -1] = 1.0

        field, metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
            spacing=(0.25, 0.6, 1.2),
        )

        expected = np.broadcast_to(
            np.linspace(0.0, 1.0, shape[0])[:, None, None], shape
        )
        np.testing.assert_allclose(field[0], expected, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(field[1:], 0.0, atol=1e-10, rtol=0.0)
        self.assertTrue(metrics.converged)

    def test_soft_observation_enters_screened_source_term(self):
        shape = (11, 9, 7)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        values = np.zeros((3, *shape), dtype=np.float64)
        soft_values = np.zeros_like(values)
        soft_values[2] = 1.0
        soft_weight = np.zeros(shape, dtype=np.float64)
        soft_weight[-1] = 20.0

        field, metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
            soft_values=soft_values,
            soft_weight=soft_weight,
        )

        self.assertTrue(metrics.converged)
        self.assertEqual(metrics.soft_voxels, shape[1] * shape[2])
        self.assertGreater(float(field[2, -1].mean()), 0.9)
        self.assertGreater(float(field[2, shape[0] // 2].mean()), 0.0)
        np.testing.assert_allclose(field[:, 0], 0.0, atol=1e-12, rtol=0.0)

    def test_normal_penalty_couples_components_toward_tangency(self):
        shape = (9, 5, 5)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        values = np.zeros((3, *shape), dtype=np.float64)
        values[0, boundary] = 1.0
        normals = np.zeros_like(values)
        normals[0, -1] = 1.0 / np.sqrt(2.0)
        normals[1, -1] = 1.0 / np.sqrt(2.0)
        penalty = np.zeros(shape, dtype=np.float64)
        penalty[-1] = 100.0

        field, metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
            normal_penalty_normals=normals,
            normal_penalty_weight=penalty,
        )

        far_dot = (field[0, -1] + field[1, -1]) / np.sqrt(2.0)
        self.assertTrue(metrics.converged)
        self.assertLess(float(np.abs(far_dot).mean()), 0.01)
        self.assertLess(float(field[1, -1].mean()), -0.4)

    def test_unanchored_component_is_rejected(self):
        shape = (8, 5, 5)
        domain = np.zeros(shape, dtype=bool)
        domain[:3] = True
        domain[5:] = True
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        values = np.zeros((3, *shape), dtype=np.float64)

        with self.assertRaisesRegex(
            ValueError, "unanchored connected components"
        ):
            _solve(
                domain_mask=domain,
                dirichlet_values=values,
                dirichlet_mask=boundary,
            )

    def test_non_convergence_is_reported_not_hidden(self):
        shape = (12, 10, 8)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        boundary[-1] = True
        values = np.zeros((3, *shape), dtype=np.float64)
        values[0, -1] = 1.0

        _, metrics = solve_weighted_screened_poisson(
            domain,
            values,
            boundary,
            device="cpu",
            dtype=torch.float64,
            tolerance=1e-14,
            max_iterations=1,
        )

        self.assertFalse(metrics.converged)
        self.assertEqual(metrics.iterations, 1)
        self.assertGreater(metrics.final_relative_residual, 1e-14)

    def test_converged_solution_is_independent_of_initial_field(self):
        shape = (10, 8, 6)
        domain = np.ones(shape, dtype=bool)
        boundary = np.zeros(shape, dtype=bool)
        boundary[0] = True
        boundary[-1] = True
        values = np.zeros((3, *shape), dtype=np.float64)
        values[0, -1] = 1.0
        random_initial = np.random.default_rng(7).normal(
            size=values.shape
        )

        zero_field, zero_metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
        )
        warm_field, warm_metrics = _solve(
            domain_mask=domain,
            dirichlet_values=values,
            dirichlet_mask=boundary,
            initial_field=random_initial,
        )

        self.assertTrue(zero_metrics.converged)
        self.assertTrue(warm_metrics.converged)
        np.testing.assert_allclose(
            zero_field, warm_field, atol=2e-8, rtol=0.0
        )


if __name__ == "__main__":
    unittest.main()
