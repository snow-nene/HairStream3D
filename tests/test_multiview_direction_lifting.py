import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.multiview_direction_lifting import lift_visible_directions


class DirectionLiftingTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((5, 5, 3), np.uint8)
        self.rgb[..., 1] = 127  # dy approximately 0, dx=+1
        self.y = np.array([1, 1, 2])
        self.x = np.array([2, 4, 4])
        self.world = np.array([[0., 0., 0.], [.02, 0., 0.],
                               [.02, 0., 0.]])

    def test_same_partition_visible_neighbor_is_lifted(self):
        result = lift_visible_directions(self.y[:2], self.x[:2], self.world[:2],
                                         np.array([1, 1]), self.rgb, self.rgb.shape[:2], step_px=2)
        self.assertTrue(result['accepted'][0])
        np.testing.assert_allclose(result['directions'][0], [1, 0, 0])

    def test_missing_or_different_layer_is_rejected(self):
        result = lift_visible_directions(self.y, self.x, self.world,
                                         np.array([1, 2, 1]), self.rgb,
                                         self.rgb.shape[:2], step_px=2)
        self.assertEqual(result['reason'][0], 'partition_or_layer_jump')
        self.assertEqual(result['reason'][1], 'neighbor_outside_image')

    def test_zero_3d_difference_is_rejected(self):
        result = lift_visible_directions(self.y[:2], self.x[:2],
                                         np.zeros((2, 3)), np.array([1, 1]),
                                         self.rgb, self.rgb.shape[:2], step_px=2)
        self.assertFalse(result['accepted'].any())


if __name__ == '__main__':
    unittest.main()
