import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.multiview_direction_lifting import compare_axial_directions


class DirectionConsistencyTests(unittest.TestCase):
    def test_sign_flip_is_axially_compatible(self):
        a=np.array([[1.,0,0],[0,1,0]])
        b=-a
        result=compare_axial_directions(a,b,max_angle_deg=1)
        self.assertTrue(result['compatible'].all())

    def test_large_turn_is_rejected(self):
        a=np.array([[1.,0,0],[1.,0,0]])
        b=np.array([[0.,1,0],[.9,.1,0]])
        result=compare_axial_directions(a,b,max_angle_deg=20)
        self.assertFalse(result['compatible'][0])
        self.assertTrue(result['compatible'][1])

    def test_zero_and_unaccepted_are_not_counted(self):
        a=np.array([[1.,0,0],[0,0,0]])
        b=np.array([[1.,0,0],[1,0,0]])
        result=compare_axial_directions(a,b,accepted=[True,False])
        self.assertEqual(result['usable'].sum(),1)


if __name__=='__main__': unittest.main()
