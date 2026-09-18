import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
from types import SimpleNamespace
from lib.surface_guide_field import SurfaceGuideField
from scripts.recon_3d.grow_visible_surface_gaps import VisibleSurfaceGrowth


def test_surface_solve_extends_tangents_and_preserves_clearance_launch():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02,resolution=12)
    points = np.array([[x,y,.021] for x in [-.005,0,.005] for y in [-.005,0,.005]])
    guides = SimpleNamespace(fields={1:(cKDTree(points),np.tile([1.,0,0],(len(points),1)))})
    chart = VisibleSurfaceGrowth(np.diag([20.,20.,20.,1.]),np.ones((64,64),int),np.zeros((64,64,2)),mesh,clearance=.003)
    field = SurfaceGuideField(mesh,guides,chart,support_m=.05)
    vector,reason = field.query(np.array([0.,0,.0208]),1)
    assert reason is None
    assert vector[0]>.8
    assert vector[2]>0
    assert field.query(np.array([0.,0,.0208]),2)[1]=='no_surface_region'


def test_surface_without_nearby_anchors_remains_unresolved():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.02,resolution=8)
    guides = SimpleNamespace(fields={1:(cKDTree(np.array([[0.,0,1.]])),np.array([[1.,0,0.]]))})
    field = SurfaceGuideField(mesh,guides,None,support_m=.05)
    assert field.fields == {}
    assert field.report['1']['status'] == 'unsupported'
    assert field.query(np.array([0.,0,.0208]),1)[1]=='no_surface_region'


def test_root_branch_selection_uses_guard_and_keeps_root():
    field=SurfaceGuideField.__new__(SurfaceGuideField)
    field.sign=1.
    field.branch_counts={'forward':0,'reverse':0}
    field.query=lambda point,partition:(np.array([field.sign,0.,0.]),None)
    root=np.array([0.,0.,0.])
    field.begin_trajectory(root,1,lambda a,b,p:'exit' if b[0]>.001 else None)
    assert field.sign==-1
    np.testing.assert_array_equal(root,np.zeros(3))
    field.begin_trajectory(root,1,lambda a,b,p:'exit' if b[0]<-.001 else None)
    assert field.sign==1
