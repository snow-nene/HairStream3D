import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from lib.multiview_fusion import compute_root_head_visibility


class RootHeadVisibilityTest(unittest.TestCase):
    def test_head_mesh_occludes_back_roots(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=24)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        mesh_path = Path(temp_dir.name) / "head.obj"
        self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_path), mesh))

        roots = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, 0.6],
            ],
            dtype=np.float32,
        )

        visible = compute_root_head_visibility(mesh_path, roots, np.eye(4))

        np.testing.assert_array_equal(visible, [True, False, True])


if __name__ == "__main__":
    unittest.main()
