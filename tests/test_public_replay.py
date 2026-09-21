import unittest
from pathlib import Path
import numpy as np
from biohub_lab.public_postprocess import load,fast_checkpoint_paths
from biohub_lab.adaptive_association import descriptors,fuse

class PublicReplayTests(unittest.TestCase):
    def test_cached_deepcenter_threshold_changes_acceptance(self):
        m=load(Path('baseline/harmonic_inference.py').read_text())
        from types import SimpleNamespace
        heat=np.full((8,8,8),.22,np.float32)
        m.deepcenter_heatmap_for_frame=lambda *args:heat
        score=m.deepcenter_score_point('sample',0,(4,16,16),{'cfg':SimpleNamespace(pool_factor=4)},{},{})
        self.assertGreater(score,.20);self.assertLess(score,.25)

    def test_fast_paths_preserve_postprocessing_code(self):
        s=Path('baseline/harmonic_inference.py').read_text();fast=fast_checkpoint_paths(s)
        # Lookup optimization must not change the graph decoder or repair logic.
        self.assertEqual(s.split('def filter_output_graph(',1)[1],fast.split('def filter_output_graph(',1)[1])
        compile(fast,'optimized','exec')

    def test_blank_border_descriptors_are_finite(self):
        nodes={1:dict(t=0,z=0,y=0,x=0),2:dict(t=0,z=7,y=31,x=31)}
        result=descriptors(nodes,np.zeros((1,8,32,32),np.float32))
        self.assertEqual(set(result),set(nodes))
        self.assertTrue(all(v.shape==(10,) and np.isfinite(v).all() for v in result.values()))

    def test_mutual_preserves_division_and_single_parent(self):
        nodes={k:dict(t=t,z=2,y=4,x=x) for k,t,x in [(1,0,4),(2,1,3),(3,1,5),(4,0,10),(5,1,10)]}
        original=[(1,2),(1,3),(4,5)];visual={(1,2):.8,(1,3):.8,(4,2):.9,(4,5):.8}
        for mode in ('mutual','adaptive','adaptive_mutual'):
            result,_=fuse(nodes,original,visual,{k:np.zeros(10) for k in nodes},mode)
            self.assertIn((1,2),result);self.assertIn((1,3),result)
            self.assertEqual(len({b for a,b in result}),len(result))
            self.assertTrue(all(nodes[b]['t']==nodes[a]['t']+1 for a,b in result))

if __name__=='__main__':unittest.main()
