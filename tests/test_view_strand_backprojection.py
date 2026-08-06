import unittest

import numpy as np

from lib.multiview_fusion import trace_view_strands_3d


class ViewStrandBackprojectionTest(unittest.TestCase):
    def test_curves_come_only_from_supplied_view_depth_and_direction(self):
        height = width = 32
        strand = np.zeros((height, width, 3), dtype=np.float32)
        strand[8:24, 4:28, 0] = 1.0
        strand[8:24, 4:28, 1] = 0.5
        strand[8:24, 4:28, 2] = 0.0
        depth = np.zeros((height, width), dtype=np.float32)
        depth[8:24, 4:28] = 0.4

        curves = trace_view_strands_3d(
            strand, depth, np.eye(4), seed_spacing=4, min_curve_points=4
        )

        self.assertGreater(len(curves), 0)
        points = np.concatenate(curves, axis=0)
        np.testing.assert_allclose(points[:, 2], 0.4, atol=1e-6)
        self.assertTrue(np.all(np.abs(np.diff(curves[0][:, 1])) < 1e-6))


if __name__ == "__main__":
    unittest.main()
