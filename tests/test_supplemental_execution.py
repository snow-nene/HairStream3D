import numpy as np

from lib.supplemental_growth import execute_supplemental_growth, plan_supplemental_growth
from lib.coverage_budget import allocate_coverage_roots


def test_new_roots_respect_each_other_spacing():
    points = np.array([[0., 0, 0], [.001, 0, 0], [1., 0, 0]])
    result = allocate_coverage_roots(points, [1, 1, 1], [1, 1, 1], [],
                                     budget=3, min_distance=.01)
    assert len(result["points"]) == 2


def test_no_guides_means_no_growth_candidates():
    plan = plan_supplemental_growth([[0, 0, 0]], [1], [True], [], [[0, 0, 0]],
                                    [], [], [], [], budget=5)
    assert len(plan["roots"]["points"]) == 0
    assert plan["stop_reason"] == "no_usable_guides"


def run_growth(guard=lambda a, b, label: None, audit=lambda p: {"passed": True},
               coverage=lambda strands: float(len(strands))):
    plan = {"roots": {"indices": [7], "points": [[0., 0, 0]], "partitions": [2]}}
    return execute_supplemental_growth(plan, query_field=lambda p, label: ([1., p[0], 0], None),
        check_segment=guard, audit_trajectory=audit, coverage_score=coverage,
        existing_strands=[], step_m=.01, length_m=.1, min_length_m=.02)


def test_field_integration_grows_curved_path_from_exact_root():
    result = run_growth()
    assert result["accepted_count"] == 1
    path = result["strands"][0]
    np.testing.assert_array_equal(path[0], [0, 0, 0])
    assert path[-1, 1] > .004
    assert result["records"][0]["partition"] == 2


def test_guard_stops_before_crossing_and_preserves_reason():
    result = run_growth(guard=lambda a, b, label: "partition_interface" if b[0] > .035 else None)
    assert result["records"][0]["stop_reason"] == "partition_interface"
    assert result["strands"][0][:, 0].max() <= .035


def test_single_point_collision_cannot_be_exported():
    result = run_growth(guard=lambda a, b, label: "head_collision")
    assert result["accepted_count"] == 0
    assert result["records"][0]["stop_reason"] == "head_collision"


def test_independent_audit_and_visible_gain_are_required():
    assert run_growth(audit=lambda p: {"passed": False})["accepted_count"] == 0
    assert run_growth(coverage=lambda strands: 0.)["accepted_count"] == 0


def test_trajectory_quality_rejects_clumped_endpoints():
    result = run_growth()
    result = execute_supplemental_growth(
        {"roots": {"indices": [1], "points": [[0., 0, 0]], "partitions": [1]}},
        query_field=lambda p, label: ([1., 0, 0], None), check_segment=lambda a, b, l: None,
        audit_trajectory=lambda p: {"passed": True}, coverage_score=lambda s: float(len(s)),
        existing_strands=[np.array([[0., 0, 0], [.001, 0, 0]])],
        trajectory_quality=lambda current, candidate: False,
        step_m=.01, length_m=.1, min_length_m=.02)
    assert result["records"][0]["rejection"] == "trajectory_quality"
