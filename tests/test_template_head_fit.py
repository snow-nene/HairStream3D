"""模板头模适配：管径间隙、原根身份和退化前缀。"""
import numpy as np
from scripts.recon_3d.fit_strands_to_template_head import fit,project_with_clearance


class Plane:
    def query(self,points):
        return points[:,2],np.tile([0.,0.,1.],(len(points),1))


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
