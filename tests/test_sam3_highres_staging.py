import unittest

import numpy as np

from scripts.infer_2d.img2masks import pad_for_sam3


class Sam3HighresStagingTest(unittest.TestCase):
    def test_keeps_native_detail_without_upscaling(self):
        image = np.full((300, 500, 3), 127, dtype=np.uint8)

        padded = pad_for_sam3(image, max_size=2048)

        self.assertEqual(padded.shape, (500, 500, 3))
        self.assertTrue(np.all(padded[100:400] == 127))
        self.assertTrue(np.all(padded[:100] == 0))
        self.assertTrue(np.all(padded[400:] == 0))

    def test_caps_large_square_input(self):
        image = np.full((2500, 1800, 3), 200, dtype=np.uint8)

        padded = pad_for_sam3(image, max_size=2048)

        self.assertEqual(padded.shape, (2048, 2048, 3))
        self.assertEqual(padded.dtype, np.uint8)

    def test_handles_odd_padding_exactly(self):
        image = np.full((5, 4, 3), 255, dtype=np.uint8)

        padded = pad_for_sam3(image, max_size=2048)

        self.assertEqual(padded.shape, (5, 5, 3))
        self.assertTrue(np.all(padded[:, :4] == 255))
        self.assertTrue(np.all(padded[:, 4] == 0))


if __name__ == "__main__":
    unittest.main()
