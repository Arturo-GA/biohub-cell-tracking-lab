import unittest
import numpy as np
import torch
from biohub_lab.visual_replay import position,replay

class DummyHead:
    def __call__(self,fa,fb,ca,cb):return -torch.cdist(ca,cb)

class ReplayTests(unittest.TestCase):
    def test_position_uses_downsample_and_relative_time(self):
        coords=np.array([[50,8,16,24]])
        p=position(coords,[100,64,256,256],0)
        self.assertEqual(p.shape,(1,32))
        np.testing.assert_array_equal(p[0,:4],np.zeros(4))
        np.testing.assert_allclose(p[0,4:8],np.ones(4))
        np.testing.assert_allclose(p[0,16],np.sin(16/256*np.pi),rtol=1e-6)
    def test_only_mutual_primary_features_and_all_candidates(self):
        nodes={0:dict(t=0,z=0,y=0,x=0),1:dict(t=0,z=0,y=0,x=10),2:dict(t=1,z=0,y=0,x=1),3:dict(t=1,z=0,y=0,x=11)}
        coords=np.array([[v[a] for a in ('t','z','y','x')] for v in nodes.values()])
        graph=dict(coords=coords,origin=np.zeros(4),shape=np.array([2,8,32,32]))
        prob,report=replay(nodes,graph,np.zeros((4,64)),DummyHead())
        self.assertEqual(len(prob),4);self.assertEqual(report['mapped_nodes'],4)
        self.assertGreater(prob[0,2],prob[1,2]);self.assertGreater(prob[1,3],prob[0,3])
        self.assertAlmostEqual(prob[0,2]+prob[1,2],1.,places=5)
if __name__=='__main__':unittest.main()
