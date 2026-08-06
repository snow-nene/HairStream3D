import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from lib.multiview_fusion import build_multiview_seg_support_volume


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

    def test_visible_background_vetoes_other_view_positive(self):
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

        self.assertFalse(support[5, 5, 10])


if __name__ == "__main__":
    unittest.main()
