"""Depth discontinuity evidence with local affine-plane rejection."""
import numpy as np
from scipy import ndimage


def depth_plane_residual(depth, valid, radii=(1, 2, 4), relative_floor=1e-5):
    """Symmetric second differences reject affine depth without fitting noise.

    Every sample in each stencil must be valid. The robust absolute residual
    retains a depth jump while a tilted plane yields zero (within roundoff).
    """
    depth = np.asarray(depth, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(depth)
    safe = np.where(valid, depth, 0)
    result = np.zeros(depth.shape, np.float32)
    for radius in radii:
        for dy, dx in ((radius, 0), (0, radius), (radius, radius), (radius, -radius)):
            a = np.roll(safe, (dy, dx), (0, 1))
            b = np.roll(safe, (-dy, -dx), (0, 1))
            usable = valid & np.roll(valid, (dy, dx), (0, 1)) & np.roll(valid, (-dy, -dx), (0, 1))
            usable[:radius] = usable[-radius:] = False
            usable[:, :radius] = usable[:, -radius:] = False
            denominator = np.maximum(np.abs(safe), 1e-6)
            residual = np.abs(a + b - 2 * safe) / denominator
            residual[(residual < relative_floor) | ~usable] = 0
            result = np.maximum(result, residual)
    result[~valid] = 0
    return result.astype(np.float32)
