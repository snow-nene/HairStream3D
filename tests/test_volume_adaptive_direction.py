"""严格体积积分缩步和组件内观测引用的回归测试。"""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.recon_strategy.volume_rk4 import adaptive_partition_rk4
from lib.recon_strategy.partition_direction import nearest_component_sources


class AdaptiveDirectionTests(unittest.TestCase):
    def test_low_support_stage_retries_with_smaller_step(self):
        def query(points):
            direction=torch.zeros_like(points);direction[0]=1
            return direction,points[0]>.2
        direction,step,failed,retries=adaptive_partition_rk4(torch.zeros((3,1)),None,query,1,.01)
        self.assertFalse(failed.any())
        self.assertEqual(step.item(),.125)
        self.assertEqual(retries.item(),3)
        self.assertAlmostEqual(direction[0,0].item(),1)

    def test_unsupported_origin_terminates_at_minimum_step(self):
        def query(points):
            return torch.zeros_like(points),torch.ones(points.shape[1],dtype=torch.bool)
        direction,step,failed,retries=adaptive_partition_rk4(torch.zeros((3,1)),None,query,1,1/64)
        self.assertTrue(failed.all())
        self.assertEqual(direction.abs().sum().item(),0)
        self.assertEqual(step.item(),1/64)

    def test_nearest_reference_stays_within_component(self):
        domain=np.ones((8,3,3),bool)
        partitions=np.ones(domain.shape,np.int32);partitions[4:]=2
        sources=np.zeros(domain.shape,bool);sources[0,1,1]=True;sources[4,1,1]=True
        nearest=nearest_component_sources(domain,partitions,sources,[1,1,1])
        self.assertEqual(nearest[0,3,1,1],0)
        self.assertEqual(nearest[0,7,1,1],4)
        sources[0,1,1]=False
        with self.assertRaisesRegex(ValueError,'no local direction source'):
            nearest_component_sources(domain,partitions,sources,[1,1,1])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
