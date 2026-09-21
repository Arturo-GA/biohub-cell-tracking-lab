import unittest
from biohub_lab.selective_repair import components,select
class SelectiveTests(unittest.TestCase):
    def test_swap_is_one_atomic_component(self):
        self.assertEqual(len(components({(1,3),(2,4)},{(1,4),(2,3)})),1)
    def test_no_context_rejects_trajectory_change(self):
        nodes={k:dict(t=t,z=1,y=1,x=x) for k,t,x in [(1,0,1),(2,0,8),(3,1,8),(4,1,1)]}
        old=[(1,3),(2,4)];new=[(1,4),(2,3)];visual={e:.1 for e in old};visual.update({e:.9 for e in new})
        edges,r=select(nodes,old,new,visual,'trajectory');self.assertEqual(edges,old)
        edges,r=select(nodes,old,new,visual,'confidence');self.assertEqual(edges,sorted(new));self.assertEqual(r['accepted'],1)
    def test_identical_graph_is_unchanged(self):
        nodes={1:dict(t=0,z=1,y=1,x=1),2:dict(t=1,z=1,y=1,x=2)}
        for mode in ('confidence','trajectory','joint'):
            edges,r=select(nodes,[(1,2)],[(1,2)],{(1,2):.8},mode);self.assertEqual(edges,[(1,2)]);self.assertEqual(r['components'],0)
if __name__=='__main__':unittest.main()
