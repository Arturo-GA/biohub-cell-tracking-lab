import unittest
import numpy as np
import torch
from biohub_lab.residual_detector import DenseCenter,TemporalDenseCenter,crop_three,positive_heatmap
from biohub_lab.residual_tracklets import augment
from biohub_lab.neural_complement import neural_chains

class ResidualTests(unittest.TestCase):
    def test_initial_temporal_model_preserves_pretrained_function(self):
        torch.set_num_threads(2);a=DenseCenter().eval();b=TemporalDenseCenter().eval();b.initialize(a.state_dict());x=torch.randn(1,3,16,16,16)
        with torch.no_grad():self.assertTrue(torch.allclose(a(x[:,1:2]),b(x),atol=1e-6))

    def test_crop_boundaries_and_positive_mask(self):
        v=np.ones((4,40,40,40));x=crop_three(v,0,[-4,4,2]);self.assertEqual(x.shape,(3,32,32,32));self.assertTrue((x==1).all())
        y=positive_heatmap(np.array([[16.,16,16]]));self.assertEqual(y[16,16,16],1);self.assertEqual(y[0,0,0],0)

    def fixture(self):
        nodes={i:dict(t=0,z=60,y=100,x=100+i) for i in range(1000)}
        nodes[1000]=dict(t=0,z=10,y=40,x=40);nodes[1001]=dict(t=4,z=10,y=40,x=40)
        c=np.array([[t,10,40,40] for t in range(5)]);p=np.ones(5)*.9;f=np.ones((5,27))/np.sqrt(27)
        return nodes,c,p,f

    def test_confirmed_gap_preserves_graph_and_adds_consecutive_edges(self):
        nodes,c,p,f=self.fixture();ns,edges,r=augment(nodes,[],c,p,f)
        self.assertEqual(r['added_nodes'],3);self.assertEqual(r['added_edges'],4)
        self.assertTrue(all(ns[k]==v for k,v in nodes.items()))
        self.assertTrue(all(ns[b]['t']==ns[a]['t']+1 for a,b in edges))

    def test_appearance_disagreement_prevents_persistence(self):
        nodes,c,p,f=self.fixture();f[2]*=-1
        ns,e,r=augment(nodes,[],c,p,f);self.assertEqual(r['added_nodes'],0)

    def test_occupied_anchor_cannot_gain_second_continuation(self):
        nodes,c,p,f=self.fixture();nodes[1002]=dict(t=1,z=60,y=100,x=100)
        ns,e,r=augment(nodes,[(1000,1002)],c,p,f);self.assertEqual(r['added_nodes'],0);self.assertEqual(e,[(1000,1002)])

    def test_unanchored_trajectory_requires_long_strong_support(self):
        nodes,_,_,_=self.fixture();c=np.array([[t,20,60,60] for t in range(10,18)]);p=np.ones(8)*.9;f=np.ones((8,27))/np.sqrt(27)
        self.assertEqual(augment(nodes,[],c,p,f,False)[2]['added_nodes'],0)
        self.assertEqual(augment(nodes,[],c,p,f,True)[2]['added_nodes'],8)
        self.assertEqual(augment(nodes,[],c,p*.6,f,True)[2]['added_nodes'],0)

    def test_neural_paths_can_override_uninformative_intensity(self):
        nodes,c,p,f=self.fixture();e=np.array([[i,i+1] for i in range(4)]);paths=neural_chains(c,e,np.ones(4)*.9)
        mapping=np.array([1000,-1,-1,-1,1001]);ns,es,r=augment(nodes,[],c,p,np.zeros_like(f),False,paths,mapping)
        self.assertEqual(r['added_nodes'],3);self.assertEqual(len(es),4)

if __name__=='__main__':unittest.main()
