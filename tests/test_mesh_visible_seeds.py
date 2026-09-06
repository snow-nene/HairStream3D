"""网格前后遮挡与相机投影回归；产物只进入 tests/outputs。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recon_3d.build_volume_partition_bundle import visible_mesh_samples


class VisibleMeshSeedTests(unittest.TestCase):
    def test_first_visible_layer_and_reprojection(self):
        output = Path(__file__).parent/'outputs/volume_partition_integration'
        output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector([
                [-2,-2,.2],[2,-2,.2],[2,2,.2],[-2,2,.2],
                [-2,-2,-.3],[2,-2,-.3],[2,2,-.3],[-2,2,-.3]])
            mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
            path = Path(temporary)/'layers.ply'
            o3d.io.write_triangle_mesh(str(path),mesh)
            seg = np.zeros((40,40),bool); seg[2:-2,2:-2] = True
            camera = np.eye(4)
            y,x,world,normals = visible_mesh_samples(path,camera,seg,return_normals=True)
            self.assertGreater(len(world),100)
            np.testing.assert_allclose(world[:,2],.2,atol=1e-6)
            np.testing.assert_allclose(world[:,0],2*x/39-1,atol=1e-6)
            np.testing.assert_allclose(world[:,1],2*y/39-1,atol=1e-6)
            np.testing.assert_allclose(np.abs(normals[:,2]),1)
            # Translate the camera: ray geometry must still project to its pixel.
            camera[0,3] = .1; camera[1,3] = -.2
            y,x,world = visible_mesh_samples(path,camera,seg)
            projected = np.c_[world,np.ones(len(world))]@camera.T
            np.testing.assert_allclose(projected[:,0],2*x/39-1,atol=1e-6)
            np.testing.assert_allclose(projected[:,1],2*y/39-1,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
