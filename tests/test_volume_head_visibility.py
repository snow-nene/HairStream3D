"""可见性诊断必须按头模深度区分前后发丝。"""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.vis.audit_volume_head_visibility import visible_raster_metrics


class HeadVisibilityTests(unittest.TestCase):
    def test_front_visible_back_hidden(self):
        output=Path(__file__).parent/'outputs/volume_partition_integration'
        output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]])
            mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
            path=Path(temporary)/'head.ply';o3d.io.write_triangle_mesh(str(path),mesh)
            strands=np.array([[[-.5,-.5,.2],[.5,-.5,.2]],
                              [[-.5,.5,-.2],[.5,.5,-.2]]])
            report,_=visible_raster_metrics(strands,np.eye(4),np.ones((21,21),bool),path)
            self.assertGreater(report['hidden_projected_pixels'],0)
            self.assertGreater(report['visible_pixels'],0)
            self.assertEqual(report['visible_leakage'],0)


if __name__=='__main__':
    unittest.main()
