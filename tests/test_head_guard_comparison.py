"""头部几何积分对照：解析平面、分区界面与固定根点回归。"""
import numpy as np
from scripts.recon_3d.compare_head_guard_integration import integrate, classify_contact


class PlaneSurface:
    def query(self, points):
        return points[:, 2], np.tile([0., 0., 1.], (len(points), 1))

    def segment_minimum(self, starts, ends, sample_m):
        return np.minimum(starts[:, 2], ends[:, 2])


def test_head_guard_and_correction_keep_roots_and_prevent_inward_growth():
    labels = np.ones((9,9,9), np.int32)
    field = np.zeros((3,9,9,9)); field[0] = 1; field[2] = -1
    roots = np.array([[0.,0.,0.]])
    common = (field, labels, roots, np.ones(1,int), np.full(3,-.1), np.full(3,.1), labels==0, labels==0, PlaneSurface())
    raw, _, _, _ = integrate(*common, 'stepwise', 8, .003)
    guarded, reason, _, _ = integrate(*common, 'head_guard', 8, .003)
    corrected, _, _, _ = integrate(*common, 'head_corrected', 8, .003)
    assert raw[0,-1,2] < -.001
    assert reason[0] == 'head_surface_collision'
    assert np.array_equal(guarded[0,0], roots[0])
    assert np.array_equal(corrected[0,0], roots[0])
    assert corrected[...,2].min() >= 0
    assert corrected[0,-1,0] > .01


def test_contact_distinguishes_interface_solid_and_unlabeled():
    labels = np.ones((3,3,3), int); labels[2] = 2
    zero = np.zeros_like(labels, bool)
    p = np.array([[1.5,1,1]])
    assert classify_contact(p, np.array([1]), labels, np.zeros(3), np.full(3,2), zero, zero) == ['partition_interface']
    labels[2] = 0
    solid = zero.copy(); solid[2] = True
    assert classify_contact(p, np.array([1]), labels, np.zeros(3), np.full(3,2), solid, zero) == ['solid_voxel']
    assert classify_contact(p, np.array([1]), labels, np.zeros(3), np.full(3,2), zero, zero) == ['unlabeled_or_outside_domain']
