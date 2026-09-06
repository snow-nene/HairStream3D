import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.direction_orientation import anchor_direction_signs, classify_multiflow_ambiguity


def test_root_anchor_resolves_axial_sign_without_changing_axis():
    directions = np.array([[1., 0, 0], [-1., 0, 0], [0., 1., 0]])
    anchors = np.array([[1., 0, 0], [1., 0, 0], [0., 0., 0]])
    result = anchor_direction_signs(directions, anchors)
    assert np.allclose(result["directions"][0], result["directions"][1])
    assert result["unresolved"][2]


def test_cross_flow_is_ambiguous_and_not_averaged():
    directions = np.array([[1., 0, 0], [0., 1, 0], [1., 0, 0]])
    result = classify_multiflow_ambiguity(directions, [1, 1, 2], angle_threshold_deg=30)
    assert result["ambiguous"][:2].all()
    assert not result["ambiguous"][2]
