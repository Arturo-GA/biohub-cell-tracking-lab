import unittest
import numpy as np
from biohub_lab.native_centers import image_offsets,apply_offsets,temporal_offsets

class NativeTests(unittest.TestCase):
    def test_flat_image_does_not_move(self):
        self.assertTrue(np.array_equal(image_offsets(np.ones((5,20,20)),np.array([[2,10,10]])),np.zeros((1,3))))
    def test_localizes_bright_center_in_native_grid(self):
        z,y,x=np.indices((9,25,25));frame=10+100*np.exp(-((z-4)**2+(y-13)**2/6+(x-14)**2/6))
        d=image_offsets(frame,np.array([[4,12,12]]))[0]
        self.assertGreater(d[1],0);self.assertGreater(d[2],d[1]);self.assertLessEqual(np.linalg.norm(d*[1.625,.40625,.40625]),1.2+1e-9)
    def test_no_identity_crossing(self):
        nodes={0:dict(t=0,z=2,y=5,x=5),1:dict(t=0,z=2,y=5,x=7)}
        ns,_=apply_offsets(nodes,[],np.array([[0,0,2],[0,0,-2]]),1.,False,(1,5,12,12));self.assertEqual(ns,nodes)
    def test_temporal_filter_reduces_isolated_offset_without_crossing_division(self):
        coords=np.zeros((5,4));edges=np.array([[0,1],[1,2],[2,3],[2,4]]);d=np.zeros((5,3));d[1,2]=2;d[3,2]=8
        result=temporal_offsets(coords,edges,d);self.assertEqual(result[1,2],1);self.assertEqual(result[3,2],8)
if __name__=='__main__':unittest.main()
