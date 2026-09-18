"""仅使用未被本轮守卫截断的局部长度证据；它仍是重建先验。"""
import numpy as np
from scipy.spatial import cKDTree


def local_length_budgets(roots, guide_roots, lengths, censored, support=.05):
    lengths = np.asarray(lengths)
    valid = (~np.asarray(censored, bool)) & np.isfinite(lengths) & (lengths >= .004) & (lengths <= .18)
    if not valid.any():
        raise ValueError('没有未截断的长度证据，不能从短前缀推断发梢')
    tree = cKDTree(np.asarray(guide_roots)[valid])
    distance, ids = tree.query(roots, k=min(8, int(valid.sum())))
    distance = np.asarray(distance).reshape(len(roots), -1)
    ids = np.asarray(ids).reshape(len(roots), -1)
    supported = distance[:, 0] <= support
    weights = np.where(distance <= support, 1/np.maximum(distance, .002)**2, 0.)
    budgets = (lengths[valid][ids]*weights).sum(1)/np.maximum(weights.sum(1), 1e-12)
    return budgets, supported
