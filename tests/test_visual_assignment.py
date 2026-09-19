import unittest
import numpy as np
from biohub_lab.visual_assignment import assign
from biohub_lab.visual_sequence import refine

class AssignmentTests(unittest.TestCase):
    def test_equivalence_to_reference_including_locked_forks(self):
        for seed in range(8):
            rng=np.random.default_rng(seed)
            nodes={i:dict(t=i//5,z=0,y=0,x=float(i%5)*8) for i in range(25)}
            old=[(i,i+5) for i in range(20)];old.remove((1,6));old.append((0,6))
            visual={(i,j):float(rng.random()) for i in range(20) for j in range((i//5+1)*5,(i//5+2)*5)}
            del visual[(7,12)]
            fast,_=assign(nodes,old,visual);slow,_=refine(nodes,old,visual,0.)
            self.assertEqual(fast,slow)
    def test_empty_candidates_preserve_graph(self):
        nodes={0:dict(t=0,z=0,y=0,x=0),1:dict(t=1,z=0,y=0,x=0)}
        self.assertEqual(assign(nodes,[(0,1)],{})[0],[(0,1)])
if __name__=='__main__':unittest.main()
