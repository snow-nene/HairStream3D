"""积分的几何守卫必须保留原始高精度体积边界。"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.recon_3d.run_pde_multiview import hair_synthesis_rk4
from lib.recon_strategy.volume_segment_guard import audit_volume_strands, constrain_volume_segments


class CanonicalBoundsTests(unittest.TestCase):
    def test_guard_receives_canonical_bounds_and_audit_passes(self):
        low = np.array([.00123456789,1.234567891,.00234567891])
        high = low+np.array([.07,.02,.02])
        labels = np.ones((8,3,3),np.int64);labels[3]=0
        field = np.zeros((3,8,3,3),np.float32);field[0]=1;field[:,3]=0
        class Field:
            _orien_vol = field
            def query_occ(self,points,calib):
                return torch.ones((1,1,points.shape[2]))
        roots = torch.tensor((low+.01).astype(np.float32))[:,None][None]
        with patch('lib.recon_strategy.volume_segment_guard.constrain_volume_segments',
                   wraps=constrain_volume_segments) as guard:
            strands = hair_synthesis_rk4(Field(),torch.device('cpu'),roots,torch.eye(4)[None],
                num_sample=12,hair_unit=.005,
                b_min_t=torch.tensor(low,dtype=torch.float32)[:,None],
                b_max_t=torch.tensor(high,dtype=torch.float32)[:,None],
                partition_label_vol=torch.tensor(labels)[None,None],
                root_partition_labels=torch.tensor([1]),volume_segment_guard=True,
                volume_bounds=(low,high))
            self.assertGreater(guard.call_count,0)
            for call in guard.call_args_list:
                np.testing.assert_array_equal(call.args[-2],low)
                np.testing.assert_array_equal(call.args[-1],high)
        self.assertTrue(audit_volume_strands(strands,[1],labels,low,high)['passed'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
