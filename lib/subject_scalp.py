"""有固定边界的头皮径向适配；保留脸部，不用任意全局缩放掩盖穿模。"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def smooth_radial_fit(vertices, faces, target_shift, movable, supported,
                      center, max_inward=.045, smoothness=3.):
    vertices = np.asarray(vertices, float)
    movable, supported = np.asarray(movable, bool), np.asarray(supported, bool)
    if max_inward <= 0 or smoothness <= 0 or not np.isfinite(target_shift).all():
        raise ValueError('invalid scalp adaptation constraints')
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                               faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = edges.T
    adjacency = sparse.coo_matrix((np.ones(len(a)*2), (np.r_[a, b], np.r_[b, a])),
                                 shape=(len(vertices),)*2).tocsr()
    lap = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    ids = np.flatnonzero(movable)
    confidence = supported.astype(float)*20
    target = np.clip(target_shift, -max_inward, 0)
    matrix = smoothness*lap[ids][:, ids] + sparse.diags(confidence[ids]+.01)
    shifts = np.zeros(len(vertices))
    shifts[ids] = spsolve(matrix, confidence[ids]*target[ids])
    radial = vertices - np.asarray(center)
    radius = np.linalg.norm(radial, axis=1)
    shifts = np.maximum(np.clip(shifts, -max_inward, 0), -.45*radius)
    result = vertices + radial/np.maximum(radius[:, None], 1e-10)*shifts[:, None]
    result[~movable] = vertices[~movable]
    return result, shifts
