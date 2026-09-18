"""诊断必须区分方向抵消、弱支持、切向投影和真正的原图冲突。"""
import numpy as np

from scripts.vis.audit_reconstruction_causes import conflict_points, interpolation_diagnostic


def test_opposite_strong_vectors_are_cancellation_not_missing_support():
    values = np.array([[1., 0, 0], [-1., 0, 0]])
    result = interpolation_diagnostic(values, np.ones(2), np.array([0., 0, 1]))
    assert result['category'] == 'saved_field_direction_cancellation'
    assert result['weighted_input_norm'] == 1
    assert result['interpolation_coherence'] == 0


def test_weak_parallel_vectors_and_zero_field_are_distinct():
    normal = np.array([0., 0, 1])
    result = interpolation_diagnostic(np.array([[.01, 0, 0], [.01, 0, 0]]), np.ones(2), normal)
    assert result['category'] == 'weak_saved_field'
    assert result['interpolation_coherence'] == 1
    assert interpolation_diagnostic(np.zeros((2, 3)), np.ones(2), normal)['category'] == 'outside_saved_support'


def test_tangent_projection_loss_is_not_direction_cancellation():
    values = np.array([[0., 0, 1], [0., 0, 1]])
    result = interpolation_diagnostic(values, np.ones(2), np.array([0., 0, 1]))
    assert result['category'] == 'tangent_projection_loss'
    assert result['before_projection_norm'] == 1
    assert result['after_projection_norm'] == 0


def test_front_conflicts_require_in_image_head_visibility_and_nonhair():
    class Front:
        hair_domain = np.array([[False, True], [False, False]])
        head_hit = np.array([[True, True], [False, True]])
        head_z = np.zeros((2, 2))

        def pixels(self, points):
            return (np.array([[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]]),
                    np.array([1., 1., 1., -1., 1.]), np.array([True, True, True, True, False]))

    conflict, distance = conflict_points(np.zeros((5, 3)), Front())
    assert conflict.tolist() == [True, False, False, False, False]
    assert np.isnan(distance[-1])
