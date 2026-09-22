import unittest
import numpy as np
from biohub_lab.trajectory_selector import fit,predict,integrate,proposals,usefulness_label

class SelectorTests(unittest.TestCase):
    def test_unknown_redundant_and_new_correct_are_distinct(self):
        links=[(0,1),(1,2)];known={(0,1):1};correct={(0,1):(100,101)}
        self.assertIsNone(usefulness_label(links,{}, {},set()))
        self.assertEqual(usefulness_label(links,known,correct,set()),1)
        self.assertEqual(usefulness_label(links,known,correct,{(100,101)}),0)
        self.assertEqual(usefulness_label(links,{**known,(1,2):0},correct,set()),0)
    def test_learns_interactions_and_roundtrips_json(self):
        import json
        rng=np.random.default_rng(77);x=rng.uniform(-1,1,(800,2));y=(x[:,0]*x[:,1]>0).astype(int)
        model=fit(x,y,np.ones(len(x)));restored=json.loads(json.dumps(model))
        test=rng.uniform(-1,1,(200,2));self.assertGreater(np.mean((predict(restored,test)>.5)==(test.prod(1)>0)),.95)
    def test_preserves_base_and_blocks_occupied_anchor(self):
        nodes={0:dict(t=0,z=0,y=0,x=0),1:dict(t=2,z=0,y=0,x=0)}
        coords=np.array([[0,0,0,0],[1,0,0,0],[2,0,0,0]]);mapping=np.array([0,-1,1]);items=[dict(whole=[0,1,2],fresh=[1])]
        ns,es,r=integrate(nodes,[],coords,mapping,items,[.9],.8)
        self.assertEqual(r['added_nodes'],1);self.assertEqual(ns[0],nodes[0]);self.assertEqual(ns[1],nodes[1]);self.assertEqual(len(es),2)
        ns,es,r=integrate(nodes,[(0,1)],coords,mapping,items,[.9],.8)
        self.assertEqual(ns,nodes);self.assertEqual(es,[(0,1)]);self.assertEqual(r['added_nodes'],0)
    def test_reject_threshold_leaves_graph_identical(self):
        nodes={0:dict(t=0,z=0,y=0,x=0)};c=np.array([[0,0,0,0],[1,0,0,0],[2,0,0,0]])
        ns,es,r=integrate(nodes,[],c,np.array([0,-1,-1]),[dict(whole=[0,1,2],fresh=[1,2])],[.99],1.01)
        self.assertEqual(ns,nodes);self.assertFalse(es);self.assertEqual(r['added_nodes'],0)
    def test_proposals_keep_low_confidence_and_high_acceleration_for_learning(self):
        c=np.array([[0,0,0,0],[1,1,0,0],[2,8,0,0],[3,9,0,0],[4,10,0,0]])
        e=np.array([[0,1],[1,2],[2,3],[3,4]]);p=proposals(c,np.array([0,-1,-1,-1,1]),np.array([1,.3,.4,.2,1]),[[0,1,2,3,4]],e,np.ones(4)*.7)
        self.assertEqual(len(p),1);self.assertEqual(p[0]['anchors'],2);self.assertEqual(len(p[0]['x']),39);self.assertTrue(np.isfinite(p[0]['x']).all())
if __name__=='__main__':unittest.main()
