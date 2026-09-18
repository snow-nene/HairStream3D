"""平滑连接调整后的根点，保留远离根部的观测 guide 主体。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def reconnect_guide(strand, root, step=.001):
    strand = np.asarray(strand, float)
    strand = strand[np.r_[True, np.linalg.norm(np.diff(strand, axis=0), axis=1) > 1e-9]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(strand, axis=0), axis=1))]
    if len(strand) < 2:
        return np.asarray(root)[None].copy()
    grid = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1]/step))+1))
    points = np.column_stack([np.interp(grid, arc, strand[:, k]) for k in range(3)])
    smoothed = gaussian_filter1d(points, 1., axis=0, mode='nearest')
    smoothed[[0, -1]] = points[[0, -1]]
    shift = np.asarray(root)-smoothed[0]
    transition = max(.015, 2*np.linalg.norm(shift))
    weight = np.maximum(0., 1-grid/transition)**2
    smoothed += weight[:, None]*shift
    smoothed[0] = root
    return smoothed
