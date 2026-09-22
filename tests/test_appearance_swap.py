import unittest
import numpy as np
from biohub_lab.appearance_swap import refine,ARMS,describe

class SwapTests(unittest.TestCase):
    def example(self):
        nodes={i:dict(t=i//2,z=2,y=5,x=5+(i%2)*2) for i in range(8)}
        edges=[(0,2),(1,3),(2,5),(3,4),(4,6),(5,7)]
        f=np.array([[float(i%2),0.] for i in range(8)])
        return nodes,edges,f
    def test_repairs_crossed_tracks_and_preserves_degrees(self):
        n,e,f=self.example();new,r=refine(n,e,f,**ARMS['strict']);self.assertIn((2,4),new);self.assertIn((3,5),new);self.assertEqual(r['accepted_swaps'],1)
    def test_identical_appearance_does_not_rewire(self):
        n,e,f=self.example();new,r=refine(n,e,np.zeros_like(f),**ARMS['appearance']);self.assertEqual(new,e)
    def test_divisions_are_not_rewired(self):
        n,e,f=self.example();e.append((2,4));new,r=refine(n,e,f,**ARMS['appearance']);self.assertEqual(new,e)
    def test_flat_image_descriptor_finite(self):
        n,_,_=self.example();f=describe(n,np.ones((4,5,12,12),np.float32));self.assertTrue(np.isfinite(f).all());self.assertTrue(np.array_equal(f,np.zeros_like(f)))
    def test_empty_graph_is_supported(self):
        f=describe({},None);new,r=refine({},[],f,**ARMS['strict']);self.assertEqual(new,[]);self.assertEqual(r['changed_edges'],0)

if __name__=='__main__':unittest.main()
