import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.coverage_budget import allocate_coverage_roots, coverage_gain


def test_coverage_budget_preserves_partitions_and_avoids_existing_roots():
    points = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]])
    result = allocate_coverage_roots(points, [1, 1, 2, 2], [1, 1, 1, 1], [[0., 0, 0]], budget=2, min_distance=.5)
    assert len(result["indices"]) == 2
    assert set(result["partitions"].tolist()) == {1, 2}
    assert 0 not in result["indices"]


def test_coverage_budget_rejects_invalid_budget():
    with pytest.raises(ValueError):
        allocate_coverage_roots(np.zeros((1, 3)), [1], [1], [], budget=-1)


def test_coverage_gain_is_explicit_and_bounded():
    assert coverage_gain([[0, 0, 0]], [[0, 0, 0], [2, 0, 0]], 0.5) == .5
