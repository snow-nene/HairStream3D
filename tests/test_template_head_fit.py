"""模板头模适配：管径间隙、原根身份和退化前缀。"""
import numpy as np
import open3d as o3d
from scripts.recon_3d.fit_strands_to_template_head import fit,project_with_clearance
from scripts.recon_3d.compare_head_guard_integration import independent_collision_audit, measure
from lib.template_identity import require_bound_template


class Plane:
    def query(self,points):
        return points[:,2],np.tile([0.,0.,1.],(len(points),1))


def test_measure_checks_single_point_prefix():
    result = measure(np.array([[[0., 0., -.002]]]), Plane())
    assert result['head_inside_1mm_roots'] == 1
    assert result['minimum_head_normal_distance_m'] == -.002


def test_independent_collision_audit_rejects_inside_sample():
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        o3d.utility.Vector3iVector([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    assert mesh.is_watertight()
    result = independent_collision_audit(np.array([[[0.1, 0.1, 0.1], [2, 2, 2]]]), mesh, 0.1)
    assert result['inside_points'] > 0
    assert not result['passed']


def test_embedded_points_move_outside_with_radius_clearance():
    p=np.array([[0,0,-.003],[0,0,.002]],dtype=float)
    result=project_with_clearance(p,Plane(),.0005)
    assert result[0,2]>=.0005
    np.testing.assert_array_equal(result[1],p[1])
    np.testing.assert_array_equal(p[:,2],[-.003,.002])


def test_all_roots_survive_and_zero_length_prefix_stays_zero():
    s=np.array([[[0,0,-.002],[.004,0,-.001],[.008,0,.002]],
                [[.02,0,-.001],[.02,0,-.001],[.02,0,-.001]]],dtype=np.float32)
    result,original,sizes=fit(s,Plane())
    assert len(result)==2 and sizes[1]==1
    assert result[:,:,2].min()>=.0005-1e-7
    assert np.linalg.norm(np.diff(result[1],axis=0),axis=1).sum()==0
    np.testing.assert_allclose(result[:,:,0],original[:,:,0],atol=1e-8)


def test_unbound_template_is_rejected(tmp_path):
    npz = tmp_path / 'strands.npz'
    np.savez(npz, strands=np.zeros((1, 2, 3), dtype=np.float32))
    blend = tmp_path / 'template.blend'
    blend.write_bytes(b'template')
    with np.load(npz) as loaded:
        try:
            require_bound_template(loaded, blend)
        except ValueError as error:
            assert 'no evaluated Blender template identity' in str(error)
        else:
            raise AssertionError('unbound strands must not pass the render gate')
