import numpy as np

from scripts.vis.audit_surface_growth_geometry import regional_metrics


def test_regions_use_target_bbox_and_keep_screen_left_separate():
    target = np.zeros((100,100),bool)
    target[60:90,70:90] = True
    covered = target.copy()
    covered[60:70,70:80] = False
    metrics = regional_metrics(target,covered)
    assert metrics['image_left']['gap_pixels'] == 100
    assert metrics['image_right']['gap_pixels'] == 0
    assert metrics['upper_third']['gap_fraction'] == .5
    assert metrics['all']['largest_gap_pixels'] == 100


def test_empty_target_has_no_fabricated_coverage():
    assert regional_metrics(np.zeros((5,5),bool),np.zeros((5,5),bool)) == {}
