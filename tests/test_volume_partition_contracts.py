"""新增模式与既有 PDE/RK4 接口的端到端行为约束。"""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.recon_strategy.partition_evidence import depth_plane_residual
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson
from lib.recon_strategy.volume_segment_guard import audit_volume_strands
from scripts.recon_3d.run_pde_multiview import hair_synthesis_rk4, query_partitioned_grid
from scripts.recon_3d.build_volume_partition_bundle import calibrate_visible_depth


class StrictVolumeBehaviorTests(unittest.TestCase):
    def test_plane_has_no_edge_but_step_does(self):
        y,x=np.mgrid[:32,:32]
        depth=1+.01*x+.02*y
        valid=np.ones(depth.shape,bool)
        self.assertLess(depth_plane_residual(depth,valid).max(),1e-6)
        depth[:,16:]+=.5
        depth[8,8]=np.nan
        evidence=depth_plane_residual(depth,valid)
        self.assertGreater(evidence[:,14:18].max(),.1)
        self.assertEqual(evidence[8,8],0)

    def test_spatial_holdout_depth_gate(self):
        y,x=np.mgrid[:64,:64]
        depth=1+.001*x+.002*y
        world=np.column_stack([x.ravel(),y.ravel(),2*depth.ravel()+.4])
        coefficient,valid,report=calibrate_visible_depth(depth,np.eye(4),y.ravel(),x.ravel(),world,.015)
        np.testing.assert_allclose(coefficient,[2,.4],atol=1e-9)
        self.assertTrue(report['passed'])
        # A spatially unsupported offset in held-out tiles must be detected.
        world[((x.ravel()//16+y.ravel()//16)%5)==0,2]+=.1
        _,_,report=calibrate_visible_depth(depth,np.eye(4),y.ravel(),x.ravel(),world,.015)
        self.assertFalse(report['passed'])

    def test_component_residuals_and_zero_rhs_are_reported(self):
        shape=(6,3,3)
        domain=np.ones(shape,bool)
        labels=np.ones(shape,np.int32);labels[3:]=2
        boundary=np.zeros(shape,bool);boundary[0]=True;boundary[-1]=True
        values=np.zeros((3,*shape));values[0,0]=1
        field,metrics=solve_weighted_screened_poisson(domain,values,boundary,
            partition_labels=labels,require_component_convergence=True,
            tolerance=1e-8,device='cpu',dtype=torch.float64)
        self.assertTrue(metrics.converged)
        self.assertEqual(len(metrics.component_residuals),2)
        self.assertIsNone(metrics.component_residuals[1]['relative_residual'])
        np.testing.assert_allclose(field[0,:3],1,atol=1e-8)
        np.testing.assert_allclose(field[:,3:],0,atol=1e-8)

    def test_strict_interpolation_does_not_extend_box(self):
        labels=torch.ones((1,1,3,3,3),dtype=torch.long)
        values,support=query_partitioned_grid(torch.ones((1,3,3,3,3)),labels,
            torch.tensor([[3.],[1.],[1.]]),torch.tensor([1]),torch.zeros(3),
            torch.ones(3)*2,strict_bounds=True)
        self.assertEqual(support.item(),0)
        self.assertEqual(values.abs().sum().item(),0)

    def test_rk4_strict_mode_stops_before_obstacle(self):
        labels=torch.ones((1,1,8,3,3),dtype=torch.long);labels[:,:,3]=0
        field=np.zeros((3,8,3,3),np.float32);field[0]=1;field[:,3]=0
        class Strategy:
            _orien_vol=field
            def query_occ(self,points,calib):
                return torch.ones((1,1,points.shape[2]))
        strands,diagnostics=hair_synthesis_rk4(Strategy(),torch.device('cpu'),
            torch.tensor([[[1.],[1.],[1.]]]),torch.eye(4).unsqueeze(0),
            num_sample=12,hair_unit=2,partition_label_vol=labels,
            root_partition_labels=torch.tensor([1]),b_min_t=torch.zeros(3,1),
            b_max_t=torch.tensor([[7.],[2.],[2.]]),volume_segment_guard=True,
            return_diagnostics=True)
        self.assertLessEqual(diagnostics['effective_hair_unit'],.5)
        self.assertTrue(audit_volume_strands(strands,[1],labels[0,0].numpy(),[0]*3,[7,2,2])['passed'])
        self.assertLess(strands[:,:,0].max(),2.5)


class CrossViewSeedTests(unittest.TestCase):
    def test_local_integer_equality_does_not_override_direction(self):
        from lib.recon_strategy.volume_partition import merge_view_partition_seeds
        labels=np.full((3,3,3),7,np.int32)
        a=np.zeros((*labels.shape,3));a[...,0]=1
        b=np.zeros_like(a);b[...,1]=1
        _,_,conflict,mapping=merge_view_partition_seeds(labels,a,labels,b,min_overlap=1)
        self.assertTrue(conflict.all())
        self.assertNotEqual(mapping['7'],7)
        _,_,conflict,mapping=merge_view_partition_seeds(labels,a,labels,-a,min_overlap=1)
        self.assertFalse(conflict.any())
        self.assertEqual(mapping['7'],7)

    def test_disjoint_view_allocates_new_identity(self):
        from lib.recon_strategy.volume_partition import merge_view_partition_seeds
        a=np.zeros((3,3,3),np.int32);a[0]=4
        b=np.zeros_like(a);b[2]=4
        directions=np.ones((*a.shape,3))
        merged,_,conflict,mapping=merge_view_partition_seeds(a,directions,b,directions,min_overlap=1)
        self.assertFalse(conflict.any())
        self.assertNotEqual(mapping['4'],4)
        self.assertTrue(np.all(merged[0]==4))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
