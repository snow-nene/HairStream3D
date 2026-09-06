"""目标分辨率几何采样与头部遮挡下的 front 约束。"""
import numpy as np
import open3d as o3d
from scripts.recon_3d.optimize_typed_hair_geometry import geometry_at_nodes,project_front_constraints


def box(size):
    mesh=o3d.geometry.TriangleMesh.create_box(size,size,size)
    mesh.translate(np.full(3,-size/2))
    return mesh


def test_fresh_target_grid_distances_have_correct_units_and_sign():
    geometry,points=geometry_at_nodes(np.full(3,-.3),np.full(3,.3),(7,7,7),box(.2),box(.4),box(.5))
    assert geometry['outer_wrap_sdf'].shape==(7,7,7)
    np.testing.assert_allclose(points[0],[-.3,-.3,-.3],atol=1e-7)
    assert geometry['outer_wrap_sdf'][3,3,3]<0
    assert geometry['outer_wrap_sdf'][6,3,3]>0
    np.testing.assert_allclose(geometry['hair_surface_distance'][6,3,3],.1,atol=1e-6)


def test_front_constraints_do_not_treat_head_occluded_points_as_visible():
    camera=np.diag([1.,-1.,1.,1.]);seg=np.ones((32,32),bool)
    strand=np.zeros((32,32,3),np.uint8);strand[...,1]=128
    p=np.array([[0.,0.,.2],[0.,0.,-.2]])
    obs,visible,_,_,_=project_front_constraints(p,camera,seg,strand,box(.2),np.tile([1.,0.,0.],(2,1)))
    assert visible.tolist()==[True,False]
    np.testing.assert_allclose(np.linalg.norm(obs,axis=1),1,atol=1e-6)


def test_termination_distinguishes_air_unknown_and_geometric_exit():
    from scripts.recon_3d.compare_head_guard_integration import classify_contact
    labels=np.ones((3,3,3),int);labels[2]=0
    empty=np.zeros_like(labels,bool);evidence=np.ones_like(labels)
    arguments=(np.array([[1.5,1,1]]),np.array([1]),labels,
               np.zeros(3),np.full(3,2),empty,empty)
    evidence[2]=3
    assert classify_contact(*arguments,evidence_class=evidence)==['air_evidence_exit']
    evidence[2]=0
    assert classify_contact(*arguments,evidence_class=evidence)==['unknown_semantics']
    outside=empty.copy();outside[2]=True
    assert classify_contact(*arguments,evidence_class=evidence,
                            outer_excluded=outside)==['outer_envelope_exit']
