import tempfile
import unittest
from pathlib import Path

from scripts.recon_3d.run_pde_multiview import resolve_front_calibration_paths


class FrontCalibrationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)
        self.param_dir = self.data_dir / "maps" / "param"
        self.param_dir.mkdir(parents=True)
        self.legacy = self.param_dir / "front.npy"
        self.dense = self.param_dir / "front_dense_silhouette.npy"
        self.legacy.touch()

    def test_legacy_front_is_the_fallback(self):
        pose, depth = resolve_front_calibration_paths(str(self.data_dir))
        self.assertEqual(Path(pose), self.legacy)
        self.assertIsNone(depth)

    def test_dense_pose_preserves_legacy_depth(self):
        self.dense.touch()
        pose, depth = resolve_front_calibration_paths(str(self.data_dir))
        self.assertEqual(Path(pose), self.dense)
        self.assertEqual(Path(depth), self.legacy)

    def test_explicit_pose_override_is_respected(self):
        self.dense.touch()
        pose, depth = resolve_front_calibration_paths(
            str(self.data_dir),
            front_calib_override="custom_pose.npy",
            front_depth_calib_override="custom_depth.npy",
        )
        self.assertEqual(pose, "custom_pose.npy")
        self.assertEqual(depth, "custom_depth.npy")


if __name__ == "__main__":
    unittest.main()
