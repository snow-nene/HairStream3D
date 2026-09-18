import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.multiview_hair_evidence import EvidenceState, classify_evidence


def test_front_visible_nonhair_is_hard_rejection():
    state, score, accepted = classify_evidence(
        front_visible=[True], front_hair=[False], side_support=[True], head_clearance=[1.])
    assert state[0] == EvidenceState.FRONT_VISIBLE_NONHAIR
    assert not accepted[0]
    assert score[0] < 0


def test_occluded_side_support_is_allowed():
    state, _, accepted = classify_evidence(
        front_visible=[False], front_hair=[False], side_support=[True], head_clearance=[1.])
    assert state[0] == EvidenceState.FRONT_OCCLUDED_SIDE_HAIR
    assert accepted[0]


def test_collision_overrides_positive_semantics():
    state, _, accepted = classify_evidence(
        front_visible=[True], front_hair=[True], side_support=[True], head_clearance=[-0.001])
    assert state[0] == EvidenceState.COLLISION
    assert not accepted[0]


def test_soft_boundary_only_adjusts_score():
    state, score, accepted = classify_evidence(
        front_visible=np.array([True, True]), front_hair=np.array([False, False]),
        side_support=np.array([False, False]), head_clearance=np.ones(2),
        boundary_distance_px=np.array([1., 8.]), soft_boundary_px=4.)
    assert np.all(state == EvidenceState.FRONT_VISIBLE_NONHAIR)
    assert not accepted.any()
    assert score[0] > score[1]
