import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.guide_quality import filter_guides


def test_guide_filter_requires_root_confidence_and_same_partition():
    guides = [np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 3))]
    result = filter_guides(guides, [1, 0, 1, 1], [[1, 1], [0, 0], [1, 2], [1]], [.9, .9, .9, .9])
    assert result["accepted_ids"] == [0]
    assert {item["reason"] for item in result["rejected"]} == {"unrooted", "cross_partition", "incomplete_trajectory"}


def test_low_confidence_guides_are_retained_as_rejections():
    result = filter_guides([np.zeros((2, 3))], [1], [[1, 1]], [.1], min_confidence=.5)
    assert result["rejected"][0]["reason"] == "low_confidence"
