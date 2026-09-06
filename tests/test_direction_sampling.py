import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.direction_sampling import sample_direction_observations


class DirectionSamplingTests(unittest.TestCase):
    def setUp(self):
        self.points=np.c_[np.arange(20),np.zeros(20),np.zeros(20)].astype(float)
        self.directions=np.tile([1.,0,0],(20,1));self.directions[10:]=[0,1,0]
        self.conf=np.ones(20);self.labels=np.r_[np.ones(10,int),np.full(10,2)]

    def test_fixed_budget_and_partition_support(self):
        r=sample_direction_observations(self.points,self.directions,self.conf,self.labels,
                                         salient_score=np.arange(20),budget=8,seed=7)
        self.assertEqual(len(r['indices']),8)
        self.assertEqual(set(self.labels[r['indices']]),{1,2})
        self.assertAlmostEqual(r['weights'].sum(),1/8*8)

    def test_sign_flip_has_same_axial_bins(self):
        a=sample_direction_observations(self.points,self.directions,self.conf,self.labels,budget=8,seed=7)
        b=sample_direction_observations(self.points,-self.directions,self.conf,self.labels,budget=8,seed=7)
        np.testing.assert_array_equal(a['direction_bins'],b['direction_bins'])
        np.testing.assert_array_equal(a['indices'],b['indices'])


if __name__=='__main__': unittest.main()
