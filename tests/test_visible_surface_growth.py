import numpy as np
import open3d as o3d

from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth, sample_polyline


def make_chart():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=40)
    camera = np.diag([20., 20., 20., 1.])
    labels = np.ones((64, 64), np.int32)
    axial = np.zeros((64, 64, 2))
    axial[..., 0] = -1
    return VisibleSurfaceGrowth(camera, labels, axial, mesh)


def test_degenerate_polylines_keep_one_point_and_chords_are_sampled():
    assert sample_polyline(np.zeros((3, 3)), .001).shape == (1, 3)
    points = sample_polyline([[0, 0, 0], [.01, 0, 0]], .001)
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= .001000001


def test_hidden_hair_does_not_count_toward_visible_coverage():
    chart = make_chart()
    hidden = np.array([[-.005, 0, -.03], [.005, 0, -.03]])
    visible = hidden.copy()
    visible[:, 2] = .03
    assert chart.score([hidden]) == 0
    assert chart.score([visible]) > 0


def test_whole_chord_and_single_point_collision_fail():
    chart = make_chart()
    assert chart.guard(np.array([0., 0, -.03]), np.array([0., 0, .03]), 1) == 'head_clearance'
    assert not chart.audit(np.array([[0., 0, 0.]]))['passed']
    assert chart.audit(np.array([[0., 0, .03], [.005, 0, .03]]))['passed']


def test_surface_query_keeps_same_chart_partition():
    chart = make_chart()
    point = np.array([0., 0, .0208])
    direction, reason = chart.query(point, 1)
    assert reason is None
    assert direction[1] > 0
    assert chart.query(point, 2)[1] == 'view_partition_boundary'


def test_perspective_template_depth_occludes_back_surface():
    near, far = .01, 1.
    projection = np.array([[3., 0, 0, 0], [0, -3., 0, 0],
        [0, 0, (far + near) / (far - near), 2 * far * near / (far - near)],
        [0, 0, -1., 0]])
    transform = np.eye(4)
    transform[2, 3] = -.1
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=40)
    chart = VisibleSurfaceGrowth(projection @ transform, np.ones((64, 64), np.int32),
                                  np.zeros((64, 64, 2)), mesh)
    assert chart.head_hit[32, 32]
    assert chart.score([np.array([[-.002, 0, -.03], [.002, 0, -.03]])]) == 0
    assert chart.score([np.array([[-.002, 0, .03], [.002, 0, .03]])]) > 0


def test_combined_audit_rejects_collision_in_existing_strand():
    from scripts.recon_3d.grow_visible_surface_gaps import audit_strands
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=20)
    result = audit_strands([np.array([[0., 0, 0]]), np.array([[0., 0, .03]])], mesh)
    assert result['strand_count'] == 2
    assert not result['passed']
    assert result['inside_points'] == 1


def test_root_launch_leaves_surface_without_moving_root():
    from lib.supplemental_growth import execute_supplemental_growth
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=40)
    labels = np.ones((64, 64), np.int32)
    axial = np.zeros((64, 64, 2))
    axial[..., 0] = -1
    chart = VisibleSurfaceGrowth(np.diag([20., 20., 20., 1.]), labels, axial, mesh, clearance=.003)
    root = np.array([0., 0, .0208])
    result = execute_supplemental_growth(
        {'roots': {'indices': [0], 'points': [root], 'partitions': [1]}},
        query_field=chart.query, check_segment=chart.guard, audit_trajectory=chart.audit,
        coverage_score=lambda strands: float(len(strands)), existing_strands=[],
        step_m=.0005, length_m=.01, min_length_m=.005)
    assert result['accepted_count'] == 1
    path = result['strands'][0]
    np.testing.assert_array_equal(path[0], root)
    assert np.linalg.norm(path[-1]) > .022


def test_export_preserves_original_samples_and_padding_exactly():
    from scripts.recon_3d.grow_visible_surface_gaps import pack_supplemental_strands
    original = np.array([[[0., 0, 0], [1e-10, 0, 0], [1e-10, 0, 0]]], np.float32)
    addition = np.array([[1., 0, 0], [2., 0, 0], [3., 0, 0], [4., 0, 0]])
    packed = pack_supplemental_strands(original, [addition])
    np.testing.assert_array_equal(packed[:1, :3], original)
    np.testing.assert_array_equal(packed[0, 3], original[0, -1])
    np.testing.assert_array_equal(packed[1], addition)


def test_template_envelope_fills_small_hair_hole_but_excludes_face():
    from scripts.recon_3d.grow_visible_surface_gaps import template_growth_domain
    pixels = np.zeros((80, 80, 3), np.uint8)
    pixels[..., 2] = 255
    pixels[10:35, 10:70] = [0, 255, 0]
    pixels[20:24, 40:44] = [0, 0, 255]
    domain = template_growth_domain(pixels, closing_radius=4)
    assert domain[22, 42]
    assert not domain[55, 40]


def test_template_guard_catches_face_between_valid_endpoints():
    from scripts.recon_3d.grow_visible_surface_gaps import guard_template_domain
    chart = make_chart()
    allowed = np.ones((64, 64), bool)
    allowed[31:34, 31:34] = False
    assert guard_template_domain(np.array([-.01, 0, .03]),
                                  np.array([.01, 0, .03]), chart, allowed) == 'template_hairline_exit'


def test_local_guides_preserve_horizontal_flow_and_reject_missing_region():
    from scripts.recon_3d.grow_visible_surface_gaps import LocalGuideField
    chart = make_chart()
    strands = [np.array([[-.01, y, .03], [0, y, .03], [.01, y, .03]]) for y in [-.002, 0, .002]]
    field = LocalGuideField(strands, chart)
    direction, reason = field.query(np.array([0., .001, .03]), 1)
    assert reason is None
    np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-6)
    assert field.query(np.array([0., 0, .03]), 2)[1] == 'no_same_region_guides'
    assert field.query(np.array([1., 0, .03]), 1)[1] == 'insufficient_local_guides'


def test_local_guides_reject_opposing_flow():
    from scripts.recon_3d.grow_visible_surface_gaps import LocalGuideField
    chart = make_chart()
    strand = np.array([[-.01, 0., .03], [0, 0, .03], [.01, 0, .03]])
    field = LocalGuideField([strand, strand[::-1]], chart)
    assert field.query(np.array([0., 0, .03]), 1)[1] == 'ambiguous_guide_flow'


def test_front_observations_use_photo_calibration(monkeypatch, tmp_path):
    import scripts.recon_3d.grow_visible_surface_gaps as growth
    import scripts.recon_3d.recon3D as recon
    calls = []
    def photo_camera(path, loadSize):
        calls.append((path, loadSize))
        return np.eye(4)
    monkeypatch.setattr(recon, 'load_calib', photo_camera)
    monkeypatch.setattr(growth, 'load_blender_view_calibration',
                        lambda *_: (_ for _ in ()).throw(AssertionError('wrong front camera')))
    np.testing.assert_array_equal(growth.load_observation_camera(tmp_path, 'front'), np.eye(4))
    assert calls == [(str(tmp_path / 'maps/param/front.npy'), 1024)]
