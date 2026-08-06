import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from lib.multiview_fusion import (
    align_vector_sign,
    build_mesh_metric_band,
    build_multiview_seg_support_volume,
)


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
