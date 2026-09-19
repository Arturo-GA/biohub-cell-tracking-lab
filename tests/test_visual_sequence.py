import unittest
from biohub_lab.visual_sequence import refine

class VisualSequenceTests(unittest.TestCase):
    def test_visual_assignment_replaces_wrong_pair(self):
        nodes={0:dict(t=0,z=0,y=0,x=0),1:dict(t=0,z=0,y=0,x=10),2:dict(t=1,z=0,y=0,x=1),3:dict(t=1,z=0,y=0,x=11)}
        out,r=refine(nodes,[(0,3),(1,2)],{(0,2):.9,(1,3):.9,(0,3):.1,(1,2):.1},0)
        self.assertEqual(out,[(0,2),(1,3)])
        self.assertGreater(r['final_energy'],r['initial_energy'])
    def test_divisions_and_uncached_edges_preserved(self):
        nodes={0:dict(t=0,z=0,y=0,x=0),1:dict(t=1,z=0,y=0,x=1),2:dict(t=1,z=0,y=0,x=2),3:dict(t=2,z=0,y=0,x=3)}
        old=[(0,1),(0,2),(1,3)]
        out,r=refine(nodes,old,{(0,1):.01,(0,2):.01,(2,3):.99})
        self.assertEqual(out,old)
    def test_sequence_resolves_ambiguous_crossing(self):
        nodes={i:dict(t=i//2,z=0,y=0,x=(i%2)*10) for i in range(8)}
        old=[(0,2),(1,3),(2,5),(3,4),(4,6),(5,7)]
        visual={e:.9 for e in old};visual.update({(2,4):.85,(3,5):.85})
        out,r=refine(nodes,old,visual)
        self.assertIn((2,4),out);self.assertIn((3,5),out)
        self.assertGreater(r['final_energy'],r['initial_energy'])
if __name__=='__main__':unittest.main()
