"""相机门禁、独立留出、遮挡和跨分区差分提升。"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.mesh_observation_geometry import audit_orthographic_camera,audit_registration,VisibleHairRaycaster,differential_visible_lift


def plane_mesh(path,z):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector([[-.02,-.02,z],[.02,-.02,z],[.02,.02,z],[-.02,.02,z]])
    mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    o3d.io.write_triangle_mesh(str(path),mesh)


class CameraGateTests(unittest.TestCase):
    def test_handedness_and_perspective_rejected(self):
        self.assertTrue(audit_orthographic_camera(np.diag([100,-100,100,1]))['passed'])
        self.assertFalse(audit_orthographic_camera(np.eye(4))['passed'])
        camera=np.diag([100.,-100.,100.,1.]);camera[3,2]=.1
        self.assertFalse(audit_orthographic_camera(camera)['passed'])

    def test_missing_and_reused_holdout_rejected(self):
        camera=np.diag([100.,-100.,100.,1.])
        self.assertFalse(audit_registration(camera,(21,21))['passed'])
        world=np.random.default_rng(42).uniform(-.005,.005,(20,3))
        pixels=((np.c_[world,np.ones(20)]@camera.T)[:,:2]+1)*10
        data={'units':np.asarray('m'),'world':world,'pixels':pixels,'heldout':np.ones(20,bool),
              'observation_ids':np.arange(20),'fit_ids':np.array([100,101])}
        self.assertTrue(audit_registration(camera,(21,21),data)['passed'])
        data['fit_ids']=np.array([0])
        self.assertEqual(audit_registration(camera,(21,21),data)['reason'],'holdout_used_in_fit')

    def test_scale_and_translation_fail_independent_correspondences(self):
        camera=np.diag([100.,-100.,100.,1.]);world=np.random.default_rng(1).uniform(-.008,.008,(20,3))
        pixels=((np.c_[world,np.ones(20)]@camera.T)[:,:2]+1)*255
        data={'units':np.asarray('m'),'world':world,'pixels':pixels,'heldout':np.ones(20,bool),
              'observation_ids':np.arange(20),'fit_ids':np.array([100])}
        shifted=camera.copy();shifted[0,3]=.1
        self.assertFalse(audit_registration(shifted,(511,511),data)['passed'])
        scaled=camera.copy();scaled[:3,:3]*=2
        self.assertFalse(audit_registration(scaled,(511,511),data)['passed'])


class DifferentialLiftTests(unittest.TestCase):
    def test_same_triangle_actual_points_and_partition_edge(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration';output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            hair=Path(temp)/'hair.ply';head=Path(temp)/'head.ply'
            plane_mesh(hair,0);plane_mesh(head,-.01)
            caster=VisibleHairRaycaster(hair,head,np.diag([100.,-100.,100.,1.]),(21,21))
            labels=np.ones((21,21),int);edge=np.zeros_like(labels,bool)
            result=differential_visible_lift(caster,[[8.,12.]],[[1.,0.]],labels,edge)
            self.assertTrue(result['accepted'][0])
            np.testing.assert_allclose(result['direction'][0],[1,0,0],atol=1e-5)
            self.assertGreater(result['distance_m'][0],.001)
            edge[:,9]=True
            result=differential_visible_lift(caster,[[8.,12.]],[[1.,0.]],labels,edge)
            self.assertTrue(result['partition_or_edge_rejected'][0])
            self.assertFalse(result['accepted'][0])

    def test_head_in_front_vetoes_hair_hit(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration'
        with tempfile.TemporaryDirectory(dir=output) as temp:
            hair=Path(temp)/'hair.ply';head=Path(temp)/'head.ply'
            plane_mesh(hair,0);plane_mesh(head,.01)
            caster=VisibleHairRaycaster(hair,head,np.diag([100.,-100.,100.,1.]),(21,21))
            hit=caster.cast_pixels([[10,10]])
            self.assertTrue(hit['hair_hit'][0]);self.assertTrue(hit['head_occluded'][0])
            self.assertFalse(hit['valid'][0])


if __name__=='__main__':unittest.main()
