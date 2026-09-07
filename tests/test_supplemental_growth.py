import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.supplemental_growth import plan_supplemental_growth


def test_supplemental_growth_reports_guide_and_random_baseline():
    points = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]])
    report = plan_supplemental_growth(points, [1, 1, 2, 2], [1, 1, 1, 1],
        np.array([[0., 0, 0]]), points, [np.zeros((2, 3))], [1], [[1, 1]], [.9],
        budget=2, radius=.1)
    # Partition 2 has no reliable guide and must not consume the budget.
    assert report["stop_reason"] == "no_visible_partition_candidates"
    assert report["roots"]["partitions"].tolist() == [1]
    assert report["guides"]["accepted_ids"] == [0]
    assert 0 <= report["coverage_gain"] <= 1
