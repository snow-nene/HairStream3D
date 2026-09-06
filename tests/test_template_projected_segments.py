"""投影后连接段应继续遵守曲面间隙，保留根数量和源对应。"""
import numpy as np
from scripts.recon_3d.fit_strands_to_template_head import fit


class Sphere:
    def query(self, points):
        radius = np.linalg.norm(points, axis=1)
        return radius - .02, points / radius[:, None]


def test_projected_segments_clear_curved_surface():
    strands = np.array([[[.001, 0, .0001], [-.001, 0, .0001]]], np.float32)
    result, reference, sizes = fit(strands, Sphere())
    p = result[0, :sizes[0]]
    midpoint = (p[:-1] + p[1:]) / 2
    assert np.linalg.norm(midpoint, axis=1).min() > .0204
    assert result.shape == reference.shape
    np.testing.assert_allclose(reference[0, 0], strands[0, 0])
    np.testing.assert_allclose(reference[0, sizes[0]-1], strands[0, -1])
