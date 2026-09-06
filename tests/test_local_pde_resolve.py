import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.local_pde_resolve import extract_local_problem, merge_local_field


def test_local_problem_keeps_halo_and_merge_does_not_overwrite_boundary():
    shape = (5, 5, 5)
    domain = np.ones(shape, bool)
    values = np.zeros((3, *shape))
    boundary = np.zeros(shape, bool); boundary[2, 2, 2] = True
    partitions = np.ones(shape, int)
    affected = np.zeros(shape, bool); affected[2, 2, 2] = True; affected[2, 2, 3] = True
    problem = extract_local_problem(domain, values, boundary, partitions, affected, halo=1)
    local = np.ones_like(problem["values"])
    merged = merge_local_field(values, local, problem)
    assert merged[:, 2, 2, 2].tolist() == [0., 0., 0.]
    assert merged[:, 2, 2, 3].tolist() == [1., 1., 1.]


def test_empty_local_region_is_rejected():
    with pytest.raises(ValueError):
        extract_local_problem(np.ones((2, 2, 2), bool), np.zeros((3, 2, 2, 2)),
                              np.zeros((2, 2, 2), bool), np.ones((2, 2, 2), int), np.zeros((2, 2, 2), bool))
