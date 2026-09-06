import json
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vis.audit_direction_lifting_ablation import run


class DirectionAblationTests(unittest.TestCase):
    def test_report_contains_all_fixed_modes(self):
        root = Path(__file__).parent / 'outputs/volume_partition_integration'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base, cache, out = Path(temp), Path(temp)/'cache', Path(temp)/'out'
            (base/'maps/strand_map').mkdir(parents=True); (base/'maps/seg').mkdir(parents=True)
            image = np.zeros((5,5,3), np.uint8); image[...,2] = 0; image[...,1] = 127
            mask = np.full((5,5),255,np.uint8)
            cv2.imwrite(str(base/'maps/strand_map/front.png'), cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base/'maps/seg/front.png'),mask)
            cache.mkdir()
            np.savez(cache/'front_calibration_samples.npz',
                     pixel_y=np.array([2,2]), pixel_x=np.array([1,3]),
                     mesh_world=np.array([[0.,0,0],[.02,0,0]]), camera=np.eye(4))
            run(type('Args',(),{'data_dir':base,'cache_dir':cache,'output_dir':out,
                                'view':'front','step_px':2,'max_angle_deg':30})())
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(set(report['modes']), {
                'mesh_tangent','visible_intersection_difference','difference_with_tangent_gate'})


if __name__=='__main__': unittest.main()
