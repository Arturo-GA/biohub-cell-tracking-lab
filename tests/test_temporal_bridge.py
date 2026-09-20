import unittest
import numpy as np
from biohub_lab.temporal_bridge import propose, augment

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.ids=np.array([10,20]); self.coords=np.array([[0,10,20,20],[3,10,20,32]],dtype=float)
        self.edges=np.empty((0,2),dtype=int)
        self.dense=np.array([[1,10,20,24],[2,10,20,28]],dtype=float)
    def test_complete_bridge_valid_and_visual_reject(self):
        p=propose(self.ids,self.coords,self.edges,self.dense); self.assertEqual(len(p),1)
        ids,c,e,r=augment(self.ids,self.coords,self.edges,self.dense,p)
        self.assertEqual(r['added_nodes'],2); self.assertEqual(len(set(ids)),4)
        self.assertEqual(e.tolist(),[[10,21],[21,22],[22,20]])
        h=np.array([[1.,0],[1,0]]); dh=np.array([[0.,1],[0,1]])
        self.assertEqual(augment(self.ids,self.coords,self.edges,self.dense,p,h,dh)[3]['bridges'],0)
    def test_missing_or_ambiguous_intermediate_rejected(self):
        self.assertEqual(propose(self.ids,self.coords,self.edges,self.dense[:1]),[])
        d=np.vstack([self.dense,[1,10,21,24]])
        self.assertEqual(propose(self.ids,self.coords,self.edges,d),[])
    def test_connected_endpoint_not_rewired(self):
        self.assertEqual(propose(self.ids,self.coords,np.array([[10,20]]),self.dense),[])

if __name__=='__main__': unittest.main()
