import numpy as np
from scripts.utils.fit_front_pose_so3 import project_landmarks, build_param
from scripts.recon_3d.recon3D import load_calib


def test_pose_fit_uses_same_pixel_coordinates_as_reconstruction(tmp_path):
    points = np.array([[.02, 1.75, .06], [-.04, 1.8, .1]])
    rotation = np.eye(3)
    center = np.array([0., 1.7, 0.])
    path = tmp_path/'camera.npy'
    np.save(path, build_param(rotation, center, 500., .2))
    matrix = load_calib(str(path)).numpy()
    projected = np.c_[points, np.ones(len(points))]@matrix.T
    expected = (projected[:, :2]+1)*511/2
    np.testing.assert_allclose(project_landmarks(points, rotation, center, 500., .2, 512),
                               expected, atol=1e-4)
