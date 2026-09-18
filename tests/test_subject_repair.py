import numpy as np
import open3d as o3d
from lib.axial_surface_field import principal_tangent
from lib.axial_surface_field import AxialSurfaceField
from lib.subject_scalp import smooth_radial_fit
from scripts.recon_3d.regrow_subject_strands import integrate, SubjectGuards
from lib.guide_reconnection import reconnect_guide
from lib.supplemental_growth import execute_supplemental_growth


def test_opposite_guides_retain_axis_and_follow_previous_step():
    directions = np.array([[1., 0, 0], [-1., 0, 0]])
    tensor = directions.T @ directions / 2
    axis, reason = principal_tangent(tensor, np.array([0., 0, 1]), np.array([-1., 0, 0]))
    assert reason is None
    np.testing.assert_allclose(axis, [-1, 0, 0])
    assert principal_tangent(np.diag([.5, .5, 0]), np.array([0., 0, 1]))[1] == 'ambiguous_axial_crossing'
    assert principal_tangent(np.zeros((3, 3)), np.array([0., 0, 1]))[1] == 'insufficient_axial_support'


def test_scalp_fit_pins_face_and_limits_inward_movement():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.1, resolution=8)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    movable = vertices[:, 2] > 0
    result, shifts = smooth_radial_fit(vertices, faces, np.full(len(vertices), -.02),
                                       movable, movable, np.zeros(3))
    np.testing.assert_array_equal(result[~movable], vertices[~movable])
    assert np.all(shifts <= 0) and np.all(shifts >= -.02)
    assert shifts[movable].min() < -.005
    mesh.vertices = o3d.utility.Vector3dVector(result)
    assert mesh.is_watertight() and not mesh.is_self_intersecting()


class StraightField:
    def query_batch(self, points, references, targets):
        return np.tile([1., 0, 0], (len(points), 1)), np.zeros(len(points), int)


class Guard:
    def points(self, points):
        return np.zeros(len(points), int)

    def segments(self, start, end):
        return np.where(end[:, 0] > .0021, 4, 0)


def test_batch_integrator_obeys_local_length_and_swept_guard():
    paths, lengths, reasons = integrate(StraightField(), Guard(), np.zeros((2, 3)),
                                        np.tile([1., 0, 0], (2, 1)), np.array([.0015, .01]), np.zeros(2))
    np.testing.assert_allclose(lengths, [.0015, .002])
    np.testing.assert_array_equal(reasons, [0, 4])
    np.testing.assert_allclose(paths[0][-1], [.0015, 0, 0])
    np.testing.assert_allclose(paths[1][-1], [.002, 0, 0])


def test_batch_integrator_handles_all_midpoints_rejected():
    class RejectMidpoint(Guard):
        def segments(self, start, end):
            assert len(start) > 0
            return np.full(len(start), 3)
    paths, lengths, reasons = integrate(StraightField(), RejectMidpoint(), np.zeros((1, 3)),
                                        np.array([[1., 0, 0]]), np.array([.01]), np.zeros(1))
    assert len(paths[0]) == 1
    assert lengths[0] == 0 and reasons[0] == 3


def test_reconnection_preserves_far_body_and_exact_root():
    original = np.column_stack([np.linspace(0, .1, 101), np.zeros((101, 2))])
    root = np.array([0, 0, -.005])
    repaired = reconnect_guide(original, root)
    np.testing.assert_array_equal(repaired[0], root)
    np.testing.assert_allclose(repaired[40:90], original[40:90], atol=1e-12)
    np.testing.assert_array_equal(repaired[-1], original[-1])


def test_supplemental_commit_only_after_guard_and_per_root_budget():
    committed = []
    result = execute_supplemental_growth(
        {'roots': {'indices': [0, 1], 'points': [[0., 0, 0], [0., 1, 0]], 'partitions': [1, 1]}},
        query_field=lambda p, label: (np.array([1., 0, 0]), None),
        check_segment=lambda a, b, label: 'boundary' if b[0] > .025 else None,
        audit_trajectory=lambda path: {'passed': True}, coverage_score=lambda paths: float(len(paths)),
        existing_strands=[], step_m=.01, length_m=.1, min_length_m=.01,
        root_length_budget=lambda root, label: .015 if root[1] == 0 else .1,
        commit_step=lambda a, b, label: committed.append((a, b)))
    np.testing.assert_allclose([r['length_m'] for r in result['records']], [.015, .02])
    assert len(committed) == 4
    assert all(end[0] <= .025 for _, end in committed)


def test_original_photo_veto_includes_points_outside_head_silhouette():
    from types import SimpleNamespace
    class Scene:
        def cast_rays(self, rays):
            return {'t_hit': o3d.core.Tensor(np.array([np.inf, 1., 3.], np.float32))}
    chart = SimpleNamespace(scene=Scene(), camera=np.eye(4), inverse=np.eye(4), head_hit=np.array([[False, True, True]]),
        head_z=np.array([[-np.inf, 1., 1.]]), hair_domain=np.array([[False, False, True]]),
        pixels=lambda points: (np.array([[0, 0], [1, 0], [2, 0]]), np.array([0., 0., 2.]), np.ones(3, bool)))
    reasons = SubjectGuards({'front': chart}).points(np.zeros((3, 3)), collision=False)
    # 无头模遮挡的非头发像素必须否决；实体后方的点与原图确认头发的点可保留。
    np.testing.assert_array_equal(reasons, [4, 0, 0])


def test_exact_front_ray_keeps_occluded_point_at_pixel_boundary():
    from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth
    mesh = o3d.geometry.TriangleMesh.create_box(width=.08, height=.08, depth=.1)
    mesh.translate([.01, -.04, -.05])
    chart = VisibleSurfaceGrowth(np.eye(4), np.zeros((9, 9), int),
                                np.zeros((9, 9, 2)), mesh)
    points = np.array([[.02, 0, -.06], [-.02, 0, -.06], [.02, 0, .06]])
    # 三点均四舍五入到 x=0 的像素，缓存射线错过位于 x>=.01 的盒子。
    assert not chart.head_hit[4, 4]
    np.testing.assert_array_equal(SubjectGuards({'front': chart}).points(points, collision=False),
                                  [0, 4, 4])


def test_side_ablation_preserves_front_veto():
    from types import SimpleNamespace
    def chart(hair, depth):
        class Scene:
            def cast_rays(self, rays):
                return {'t_hit': o3d.core.Tensor(np.full(len(rays), 1. if depth < 1 else 3., np.float32))}
        return SimpleNamespace(scene=Scene(), camera=np.eye(4), inverse=np.eye(4), head_hit=np.ones((1,1),bool), head_z=np.ones((1,1)),
            hair_domain=np.array([[hair]]), pixels=lambda points:
            (np.zeros((len(points),2),int), np.full(len(points),depth), np.ones(len(points),bool)))
    charts = {'front':chart(False,0), 'left':chart(False,2), 'right':chart(False,2)}
    points = np.zeros((1,3))
    assert SubjectGuards(charts).points(points,collision=False)[0] == 6
    assert SubjectGuards(charts,side_veto=False).points(points,collision=False)[0] == 0
    charts['front'] = chart(False,2)
    assert SubjectGuards(charts,side_veto=False).points(points,collision=False)[0] == 4


def test_solved_axial_field_preserves_opposite_guides_and_branch_continuity():
    from types import SimpleNamespace
    from scipy.spatial import cKDTree
    from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=10)
    samples = np.array([[-.001, 0., .02], [.001, 0., .02]])
    guides = SimpleNamespace(fields={1: (cKDTree(samples), np.array([[1., 0, 0], [-1., 0, 0]]))})
    chart = VisibleSurfaceGrowth(np.diag([20., 20., 20., 1.]), np.ones((32, 32), int),
                                 np.zeros((32, 32, 2)), mesh)
    field = AxialSurfaceField(mesh, guides, chart)
    directions, reasons = field.query_batch(np.array([[0., 0, .021], [0., 0, .021]]),
        np.array([[1., 0, 0], [-1., 0, 0]]), np.array([.001, .001]))
    np.testing.assert_array_equal(reasons, [0, 0])
    assert directions[0, 0] > .9 and directions[1, 0] < -.9
    assert field.report['1']['linear_relative_residual'] < 1e-8
