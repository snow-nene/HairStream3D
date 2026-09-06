import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.visibility_metrics import visible_projection_metrics


def test_visible_metrics_are_axial_and_report_leakage():
    uv = np.array([[1, 1], [3, 1], [-1, 0]], float)
    pred = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0]], float)
    target = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], float)
    result = visible_projection_metrics(uv, pred, target, [1, 1, 1], np.ones((2, 4), bool))
    assert result["visible_samples"] == 2
    assert result["axial_angle_median_deg"] == 0.0
    assert result["out_of_bounds"] == 1


def test_visible_metrics_do_not_count_nonhair_as_coverage():
    result = visible_projection_metrics([[0, 0]], [[1, 0, 0]], [[1, 0, 0]], [1], np.zeros((2, 2), bool))
    assert result["leakage_rate"] == 1.0
