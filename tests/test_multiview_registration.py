import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.multiview_registration import registration_report, gate_registration


class RegistrationTests(unittest.TestCase):
    def test_exact_orthographic_projection_passes(self):
        camera = np.array([[127.5,0,0,127.5],[0,127.5,0,127.5],[0,0,1,0],[0,0,0,1.]])
        world = np.array([[.1, -.2, 1.], [-.3, .4, 1.]])
        pixels = world[:, :2] @ camera[:2,:2].T + camera[:2,3]
        report = registration_report(world, pixels, camera, (256, 256))
        self.assertTrue(report["passed"])
        gate_registration(world, pixels, camera, (256, 256))

    def test_scale_or_axis_error_fails(self):
        camera = np.array([[127.5,0,0,127.5],[0,127.5,0,127.5],[0,0,1,0],[0,0,0,1.]])
        world = np.array([[.1, -.2, 1.], [-.3, .4, 1.]])
        pixels = world[:, :2] @ camera[:2,:2].T + camera[:2,3]
        pixels[:, 1] = 255 - pixels[:, 1]
        report = registration_report(world, pixels, camera, (256, 256),
                                     max_error_px=.5)
        self.assertFalse(report["passed"])
        with self.assertRaises(ValueError):
            gate_registration(world, pixels, camera, (256, 256), max_error_px=.5)


if __name__ == "__main__":
    unittest.main()
