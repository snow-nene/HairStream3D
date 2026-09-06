import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.manual_curve_constraints import validate_manual_curves, affected_region_mask


def test_manual_curve_rejects_cross_partition_and_occlusion():
    curves = [np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))]
    pixels = [np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 0]])]
    visible = np.ones((2, 2), bool)
    labels = np.ones((2, 2), int)
    labels[0, 1] = 2
    visible[0, 1] = False
    result = validate_manual_curves(curves, pixels, visible, labels)
    assert [item["reason"] for item in result["rejected"]] == ["occluded", "occluded", "occluded"]
    visible[0, 1] = True
    result = validate_manual_curves(curves[:1], pixels[:1], visible, labels)
    assert result["rejected"][0]["reason"] == "cross_partition"


def test_manual_curve_accepts_same_visible_partition():
    result = validate_manual_curves([np.zeros((2, 3))], [np.array([[0, 0], [1, 0]])],
                                    np.ones((1, 2), bool), np.ones((1, 2), int))
    assert len(result["accepted"]) == 1
    assert result["automatic_path_unchanged"]


def test_accepted_curve_generates_local_resolve_region():
    mask = affected_region_mask([np.array([[1., 1., 1.]])], (4, 4, 4), np.eye(4), radius_voxels=1)
    assert mask.sum() == 27
    assert not mask[3, 3, 3]
