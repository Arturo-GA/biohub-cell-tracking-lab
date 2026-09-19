import unittest
import numpy as np
from biohub_lab.visual_candidates import decode_cache

class CandidateTests(unittest.TestCase):
    def test_smoothing_does_not_break_identity(self):
        raw={15:dict(t=0,z=1,y=2,x=3),40:dict(t=1,z=1,y=2,x=4)}
        final={15:dict(t=0,z=1,y=2,x=5),40:dict(t=1,z=1,y=2,x=6)}
        frames=[dict(source_coords=np.array([[0,1,2,3]]),target_coords=np.array([[1,1,2,4]]),edges=np.array([[0,0]]),prob=np.array([.9]))]
        p,r=decode_cache(raw,final,frames);self.assertAlmostEqual(p[15,40],.9);self.assertEqual(r['mapped_pairs'],1)
        final[15]['t']=1
        with self.assertRaises(AssertionError):decode_cache(raw,final,frames)
if __name__=='__main__':unittest.main()
