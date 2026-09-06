"""根连接不能通过实体、可信空气、冲突或其他分区补齐根集合。"""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.recon_strategy.volume_root_connection import plan_root_connections, audit_root_connections


class RootConnectionTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.zeros((9,5,5),np.int64);self.labels[4:]=1
        self.solid=np.zeros_like(self.labels,bool)
        self.evidence=np.zeros_like(self.labels,np.uint8)
        self.conflict=np.zeros_like(self.labels,bool)
        self.low=np.zeros(3);self.high=np.array([.08,.04,.04])

    def plan(self,roots,**kwargs):
        return plan_root_connections(np.array(roots,dtype=np.float32),self.labels,self.solid,
            self.evidence,self.conflict,self.low,self.high,**kwargs)

    def test_inside_and_short_connector_preserve_original_identity(self):
        roots=[[.05,.02,.02],[.03,.02,.02],[.0,.02,.02]]
        p=self.plan(roots)
        self.assertEqual(p['status'].tolist(),['inside','short_connection','too_far'])
        np.testing.assert_array_equal(p['roots_world'],np.asarray(roots,np.float32))
        np.testing.assert_array_equal(p['root_labels'],[1,1,0])
        self.assertTrue(audit_root_connections(p,self.labels,self.solid,self.evidence,
            self.conflict,self.low,self.high)['passed'])

    def test_thin_solid_wall_blocks_connection(self):
        self.solid[3]=True
        p=self.plan([[.02,.02,.02]],max_distance=.035)
        self.assertEqual(p['status'][0],'blocked_connector')

    def test_air_and_conflict_are_not_unknown_corridors(self):
        self.evidence[3]=3
        self.assertEqual(self.plan([[.03,.02,.02]])['status'][0],'blocked_connector')
        self.evidence[:]=0;self.conflict[3]=True
        self.assertEqual(self.plan([[.03,.02,.02]])['status'][0],'blocked_connector')

    def test_equally_near_partitions_reject_assignment(self):
        self.labels[4:,:2]=2
        p=self.plan([[.03,.015,.02]])
        self.assertEqual(p['status'][0],'ambiguous_partition')

    def test_audit_rejects_modified_connector(self):
        p=self.plan([[.03,.02,.02]])
        p['starts_world'][0]=[.06,.02,.02]
        self.assertFalse(audit_root_connections(p,self.labels,self.solid,self.evidence,
            self.conflict,self.low,self.high)['passed'])


if __name__=='__main__':
    unittest.main()
