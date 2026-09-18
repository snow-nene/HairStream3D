import numpy as np
import pytest
from lib.guide_length_evidence import local_length_budgets


def test_nearest_cut_prefix_does_not_shorten_budget():
    budget, supported = local_length_budgets([[0.,0,0]], [[0.,0,0],[.01,0,0]], [.005,.08], [True,False])
    np.testing.assert_allclose(budget, [.08])
    assert supported[0]


def test_remote_evidence_does_not_supply_length():
    budget, supported = local_length_budgets([[0.,0,0]], [[1.,0,0]], [.08], [False])
    assert budget[0] == 0 and not supported[0]


def test_all_censored_is_explicit_failure():
    with pytest.raises(ValueError):
        local_length_budgets([[0.,0,0]], [[0.,0,0]], [.01], [True])
