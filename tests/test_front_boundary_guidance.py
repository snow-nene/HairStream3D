import numpy as np
import open3d as o3d
from lib.front_boundary_guidance import FrontBoundaryGuidance
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth


def test_guidance_is_bounded_and_leaves_hidden_directions_unchanged():
    mesh = o3d.geometry.TriangleMesh.create_box(.2,.2,.1)
    mesh.translate([-.1,-.1,-.05])
    mask = np.zeros((101,101), int)
    mask[:,50:] = 1
    chart = VisibleSurfaceGrowth(np.eye(4),mask,np.zeros((101,101,2)),mesh)
    guidance = FrontBoundaryGuidance(chart)
    points = np.array([[.02,0,.06],[.02,0,-.06],[.2,0,.06]])
    directions = np.tile([-.6,-.8,0.],(3,1))
    normals = np.tile([0.,0,1.],(3,1))
    result = guidance(points,directions,normals)
    assert result[0,0] > directions[0,0]
    np.testing.assert_allclose(result[1:],directions[1:],atol=1e-12)
    assert np.degrees(np.arccos(np.clip((result*directions).sum(1),-1,1))).max() <= 15.01
    np.testing.assert_allclose(np.linalg.norm(result,axis=1),1)
