import unittest
import numpy as np
from biohub_lab.weighted_bridge import combine

class WeightedBridgeTests(unittest.TestCase):
    def setUp(self):
        self.ids=np.array([10,20]);self.coords=np.array([[0,10,20,20],[3,10,20,32]])
        self.edges=np.empty((0,2),int);self.dense=np.array([[1,10,20,24],[2,10,20,28]])

    def test_agreement_preserves_observed_nodes_and_consecutive_topology(self):
        ids,coords,edges,report=combine(self.ids,self.coords,self.edges,self.dense,self.dense,'weighted')
        np.testing.assert_array_equal(coords[:2],self.coords)
        self.assertEqual(report['bridges'],1)
        times=dict(zip(ids,coords[:,0]))
        self.assertTrue(all(times[b]==times[a]+1 for a,b in edges))
        self.assertEqual(len(set(ids)),len(ids))

    def test_remote_expert_cannot_move_a_dense_detection(self):
        a=combine(self.ids,self.coords,self.edges,self.dense,self.dense+np.array([0,0,0,100]),'weighted')
        b=combine(self.ids,self.coords,self.edges,self.dense,self.dense,'dense')
        np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        self.assertEqual(a[3]['mean_secondary_weight'],0)

    def test_no_motion_evidence_does_not_hallucinate_flow(self):
        result=combine(self.ids,self.coords,self.edges,self.dense,self.dense,'weighted_motion')
        self.assertEqual(result[3]['bridges'],0)

if __name__=='__main__':unittest.main()
