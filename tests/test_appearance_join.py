import unittest
import numpy as np
from biohub_lab.appearance_join import refine,ARMS

class JoinTests(unittest.TestCase):
    def sample(self):
        n={i:dict(t=i,z=2,y=5,x=5+i) for i in range(6)};e=[(0,1),(1,2),(3,4),(4,5)];f=np.zeros((6,3));return n,e,f
    def test_joins_consistent_fragments(self):
        n,e,f=self.sample();new,r=refine(n,e,f,**ARMS['strict']);self.assertEqual(new,e+[(2,3)]);self.assertEqual(r['added_edges'],1)
    def test_rejects_appearance_disagreement(self):
        n,e,f=self.sample();f[3:]=10;new,_=refine(n,e,f,**ARMS['recall']);self.assertEqual(new,e)
    def test_never_replaces_an_existing_link(self):
        n,e,f=self.sample();e.append((2,3));new,_=refine(n,e,f,**ARMS['recall']);self.assertEqual(new,e)
    def test_empty_graph(self):
        self.assertEqual(refine({},[],np.empty((0,3)),**ARMS['strict'])[0],[])

if __name__=='__main__':unittest.main()
