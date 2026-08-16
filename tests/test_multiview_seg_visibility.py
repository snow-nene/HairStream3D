import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from lib.multiview_fusion import (
    align_vector_sign,
    build_mesh_root_guidance,
    build_mesh_metric_band,
    build_multiview_seg_support_volume,
    limit_direction_normal_component,
    orient_sparse_direction_axes,
)
from lib.silhouette_guard import STATUS_HARD, STATUS_OK, SilhouetteGuard


def _write_head_sphere(path):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=12)
    assert o3d.io.write_triangle_mesh(str(path), mesh)


class MultiviewSegVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.head_path = Path(self.temp_dir.name) / "head.obj"
        _write_head_sphere(self.head_path)
        self.calib = np.eye(4, dtype=np.float32)
        self.b_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        self.b_max = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def build_support(self, calibs, masks):
        return build_multiview_seg_support_volume(
            calibs,
            masks,
            self.b_min,
            self.b_max,
            resolution=11,
            slab_size=3,
            head_mesh_path=str(self.head_path),
            visibility_tolerance=0.0,
        )

    def build_stub_silhouette_guard(self, primary_authoritative):
        guard = SilhouetteGuard.__new__(SilhouetteGuard)
        guard.hard_px = 20.0
        guard.views = ["front", "left"]
        guard.primary = "front"
        guard.side_views = ["left"]
        guard.primary_authoritative = primary_authoritative
        guard._project_px = lambda view, points: (
            np.zeros(len(points), dtype=np.float32),
            np.zeros(len(points), dtype=np.float32),
        )
        guard._sample_sd = lambda view, px, py: (
            np.full(
                len(px),
                -30.0 if view == "front" else 5.0,
                dtype=np.float32,
            ),
            np.ones(len(px), dtype=bool),
        )
        guard._visible = lambda view, points: np.ones(len(points), dtype=bool)
        return guard

    def test_primary_silhouette_mode_keeps_front_veto(self):
        guard = self.build_stub_silhouette_guard(primary_authoritative=True)
        status = guard.classify(np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(status.tolist(), [STATUS_HARD])

    def test_union_silhouette_mode_accepts_visible_side_support(self):
        guard = self.build_stub_silhouette_guard(primary_authoritative=False)
        status = guard.classify(np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(status.tolist(), [STATUS_OK])

    def test_head_occlusion_abstains_and_visible_seg_supports(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2:7] = True
        support = self.build_support(
            {"front": (self.calib, np.eye(3, dtype=np.float32))},
            {"front": mask},
        )

        self.assertTrue(support[5, 5, 10])
        self.assertFalse(support[5, 5, 0])

    def test_front_positive_overrides_side_background(self):
        positive = np.ones((9, 9), dtype=bool)
        negative = np.zeros((9, 9), dtype=bool)
        calibs = {
            "front": (self.calib, np.eye(3, dtype=np.float32)),
            "side": (self.calib, np.eye(3, dtype=np.float32)),
        }
        support = self.build_support(
            calibs,
            {"front": positive, "side": negative},
        )

        self.assertTrue(support[5, 5, 10])

    def test_front_background_vetoes_side_positive(self):
        positive = np.ones((9, 9), dtype=bool)
        negative = np.zeros((9, 9), dtype=bool)
        calibs = {
            "front": (self.calib, np.eye(3, dtype=np.float32)),
            "side": (self.calib, np.eye(3, dtype=np.float32)),
        }
        support = self.build_support(
            calibs,
            {"front": negative, "side": positive},
        )

        self.assertFalse(support[5, 5, 10])

    def test_side_tie_keeps_positive_when_front_is_unavailable(self):
        positive = np.ones((9, 9), dtype=bool)
        negative = np.zeros((9, 9), dtype=bool)
        calibs = {
            "left": (self.calib, np.eye(3, dtype=np.float32)),
            "right": (self.calib, np.eye(3, dtype=np.float32)),
        }
        support = self.build_support(
            calibs,
            {"left": positive, "right": negative},
        )

        self.assertTrue(support[5, 5, 10])

    def test_metric_band_uses_world_space_distance_and_support(self):
        support = np.ones((21, 21, 21), dtype=bool)
        support[10, 10, 16] = False
        band = build_mesh_metric_band(
            self.head_path,
            support,
            self.b_min,
            self.b_max,
            band_width=0.06,
            slab_size=4,
        )

        self.assertFalse(band[10, 10, 10])
        self.assertFalse(band[10, 10, 20])
        self.assertFalse(band[10, 10, 16])
        self.assertTrue(band[10, 10, 15])

    def test_vector_sign_alignment_preserves_reference_hemisphere(self):
        import torch

        reference = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
        )
        vectors = torch.tensor(
            [[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
        )
        aligned = align_vector_sign(vectors, reference)

        self.assertTrue(torch.all(torch.sum(aligned * reference, dim=0) >= 0.0))

    def test_mesh_root_guidance_prefers_authoritative_seg(self):
        roots = np.array([[0.0, 0.0, 0.3]], dtype=np.float32)
        strand = np.ones((9, 9, 3), dtype=np.float32)
        seg = np.zeros((9, 9), dtype=bool)
        guidance = build_mesh_root_guidance(
            self.head_path,
            roots,
            {"front": (self.calib, np.eye(3, dtype=np.float32))},
            {"front": strand},
            seg_masks={"front": seg},
        )

        self.assertFalse(guidance["visible_roots"]["front"].any())

    def test_projection_mesh_support_does_not_require_visible_nearest_root(self):
        # Mesh support and root launch visibility are separate concepts: a
        # visible hair tip may be nearest to a scalp root hidden by the head.
        # This invariant is exercised by keeping seg as the root gate above;
        # projection rendering must not mutate or further shrink that gate.
        roots = np.array([[0.0, 0.0, -0.3]], dtype=np.float32)
        strand = np.ones((9, 9, 3), dtype=np.float32)
        seg = np.ones((9, 9), dtype=bool)
        guidance = build_mesh_root_guidance(
            self.head_path,
            roots,
            {"front": (self.calib, np.eye(3, dtype=np.float32))},
            {"front": strand},
            seg_masks={"front": seg},
            head_mesh_path=str(self.head_path),
        )

        self.assertFalse(guidance["visible_roots"]["front"][0])
        self.assertGreater(len(guidance["vertices"]), 0)

    def test_sparse_direction_axes_propagate_consistent_sign(self):
        shape = (12, 12, 12)
        coordinates = np.array(
            [[5, 5, 2], [5, 5, 4], [5, 5, 6], [5, 5, 8]]
        )
        flat_ids = np.ravel_multi_index(coordinates.T, shape)
        axes = np.array(
            [[0, -1, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        references = np.array(
            [[0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]],
            dtype=np.float32,
        )

        oriented = orient_sparse_direction_axes(
            flat_ids,
            axes,
            references,
            np.ones(4, dtype=np.int8),
            shape,
        )

        self.assertTrue(np.all(oriented[:, 1] < 0.0))
        self.assertTrue(np.all(np.sum(oriented[:-1] * oriented[1:], axis=1) > 0.99))

    def test_side_direction_axes_use_downward_component_anchor(self):
        shape = (12, 12, 12)
        coordinates = np.array([[5, 5, 2], [5, 5, 4], [5, 5, 6]])
        flat_ids = np.ravel_multi_index(coordinates.T, shape)
        upward_axes = np.tile(
            np.array([[0.1, 0.99, 0.0]], dtype=np.float32), (3, 1)
        )

        oriented = orient_sparse_direction_axes(
            flat_ids,
            upward_axes,
            upward_axes,
            np.ones(3, dtype=np.int8),
            shape,
            preferred_owners=[1],
        )

        self.assertTrue(np.all(oriented[:, 1] < 0.0))

    def test_front_direction_axes_keep_original_component_anchor(self):
        shape = (12, 12, 12)
        coordinates = np.array([[5, 5, 2], [5, 5, 4]])
        flat_ids = np.ravel_multi_index(coordinates.T, shape)
        upward_axes = np.tile(
            np.array([[0.1, 0.99, 0.0]], dtype=np.float32), (2, 1)
        )

        oriented = orient_sparse_direction_axes(
            flat_ids,
            upward_axes,
            upward_axes,
            np.zeros(2, dtype=np.int8),
            shape,
            preferred_owners=[1],
        )

        self.assertTrue(np.all(oriented[:, 1] > 0.0))

    def test_direction_normal_component_is_clamped_without_losing_sign(self):
        directions = np.array(
            [[0.6, -0.8, 0.0], [-0.6, -0.8, 0.0], [0.2, -0.98, 0.0]],
            dtype=np.float32,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        normals = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))

        limited, changed = limit_direction_normal_component(
            directions, normals, max_component=0.3
        )

        self.assertTrue(np.array_equal(changed, [True, True, False]))
        self.assertTrue(np.all(np.abs(limited[:, 0]) <= 0.30001))
        self.assertTrue(np.all(limited[:, 1] < 0.0))
        self.assertTrue(np.allclose(np.linalg.norm(limited, axis=1), 1.0))

    def test_hair_surface_occlusion_makes_front_background_abstain(self):
        distant_head = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.1, resolution=12
        )
        distant_head.translate((0.0, 5.0, 0.0))
        head_path = Path(self.temp_dir.name) / "distant_head.obj"
        self.assertTrue(o3d.io.write_triangle_mesh(str(head_path), distant_head))

        occluder = o3d.geometry.TriangleMesh.create_box(
            width=1.0, height=1.0, depth=0.02
        )
        occluder.translate((-0.5, -0.5, 0.3))
        occluder_path = Path(self.temp_dir.name) / "hair_occluder.obj"
        self.assertTrue(
            o3d.io.write_triangle_mesh(str(occluder_path), occluder)
        )

        front_calib = np.eye(4, dtype=np.float32)
        back_calib = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        masks = {
            "front": np.zeros((9, 9), dtype=bool),
            "back": np.ones((9, 9), dtype=bool),
        }
        calibs = {
            "front": (front_calib, np.eye(3, dtype=np.float32)),
            "back": (back_calib, np.eye(3, dtype=np.float32)),
        }
        support = build_multiview_seg_support_volume(
            calibs,
            masks,
            self.b_min,
            self.b_max,
            resolution=11,
            slab_size=3,
            head_mesh_path=str(head_path),
            occluder_mesh_path=str(occluder_path),
            visibility_tolerance=0.0,
        )

        self.assertTrue(support[5, 5, 5])


if __name__ == "__main__":
    unittest.main()
