import numpy as np
from lib.coverage_acceptance import accept_coverage_candidates


class Chart:
    target = np.array([[True, True, False]])

    def covered(self, paths):
        return np.array([[True, False, False]])

    def mask(self, path):
        return np.asarray(path, int)


def test_rejects_existing_background_and_duplicate_new_coverage():
    accepted, gains = accept_coverage_candidates([[0], [2], [1], [1]], [.01]*4,
                                                 {'front': Chart()}, [])
    np.testing.assert_array_equal(accepted, [False, False, True, False])
    np.testing.assert_array_equal(gains, [0, 0, 1, 0])


def test_short_candidate_does_not_consume_coverage():
    accepted, gains = accept_coverage_candidates([[1], [1]], [.001, .01], {'front': Chart()}, [])
    np.testing.assert_array_equal(accepted, [False, True])
