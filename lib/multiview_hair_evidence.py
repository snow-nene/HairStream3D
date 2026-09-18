"""多视角头发候选的可见性证据状态与连续评分。"""
from enum import IntEnum
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class EvidenceState(IntEnum):
    UNKNOWN = 0
    FRONT_VISIBLE_HAIR = 1
    FRONT_VISIBLE_NONHAIR = 2
    FRONT_OCCLUDED_SIDE_HAIR = 3
    SIDE_ONLY = 4
    COLLISION = 5


def classify_evidence(*, front_visible, front_hair, side_support,
                      head_clearance, boundary_distance_px=99.,
                      soft_boundary_px=0.):
    """按统一优先级分类候选点。

    front_visible_nonhair 永远硬拒绝；front 被遮挡时允许侧视支持。
    soft_boundary 只降低惩罚，不把明确的 front 非头发变成头发真值。
    """
    front_visible = np.asarray(front_visible, bool)
    front_hair = np.asarray(front_hair, bool)
    side_support = np.asarray(side_support, bool)
    clearance = np.asarray(head_clearance, float)
    boundary_distance_px = np.broadcast_to(np.asarray(boundary_distance_px, float), front_visible.shape)
    near_boundary = boundary_distance_px <= float(soft_boundary_px)
    if not (front_visible.shape == front_hair.shape == side_support.shape == clearance.shape == near_boundary.shape):
        raise ValueError('evidence arrays must have identical shape')
    state = np.full(front_visible.shape, EvidenceState.UNKNOWN, np.int8)
    state[front_visible & ~front_hair] = EvidenceState.FRONT_VISIBLE_NONHAIR
    state[front_visible & front_hair] = EvidenceState.FRONT_VISIBLE_HAIR
    occluded = ~front_visible
    state[occluded & side_support] = EvidenceState.FRONT_OCCLUDED_SIDE_HAIR
    state[~front_visible & ~side_support] = EvidenceState.SIDE_ONLY
    state[clearance < 0.] = EvidenceState.COLLISION
    score = np.zeros(front_visible.shape, np.float32)
    score[state == EvidenceState.FRONT_VISIBLE_HAIR] = 3.
    score[state == EvidenceState.FRONT_OCCLUDED_SIDE_HAIR] = 2.
    score[state == EvidenceState.SIDE_ONLY] = .5
    score[state == EvidenceState.FRONT_VISIBLE_NONHAIR] = -4.
    score[state == EvidenceState.COLLISION] = -5.
    if soft_boundary_px > 0:
        near = near_boundary & (state == EvidenceState.FRONT_VISIBLE_NONHAIR)
        score[near] += 2. * np.clip(1. - boundary_distance_px[near] / soft_boundary_px, 0., 1.)
    accepted = np.isin(state, [EvidenceState.FRONT_VISIBLE_HAIR,
                               EvidenceState.FRONT_OCCLUDED_SIDE_HAIR,
                               EvidenceState.SIDE_ONLY]) & (clearance >= 0.)
    return state, score, accepted


def diffuse_surface_evidence(vertices, edges, anchors, values, smoothness=20.):
    """在网格图上扩散局部证据，返回连续分数；不改变几何。"""
    vertices = np.asarray(vertices)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    anchors = np.asarray(anchors, dtype=np.int64).reshape(-1)
    values = np.asarray(values, float).reshape(-1)
    if len(anchors) != len(values) or np.any(anchors < 0) or np.any(anchors >= len(vertices)):
        raise ValueError('invalid evidence anchors')
    if len(edges):
        w = np.ones(len(edges))
        adj = sparse.coo_matrix((np.r_[w,w], (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])), shape=(len(vertices),len(vertices))).tocsr()
        lap = sparse.diags(np.asarray(adj.sum(1)).ravel()) - adj
    else:
        lap = sparse.csr_matrix((len(vertices),len(vertices)))
    confidence = np.zeros(len(vertices)); confidence[anchors] = 1.
    rhs = confidence * 0.
    rhs[anchors] = values
    system = smoothness * lap + sparse.diags(confidence + 1e-6)
    return spsolve(system, rhs)
