import unittest
import numpy as np
import torch
from biohub_lab.association_rank import (AssociationRanker, competing_groups,
    sample_rows, ranking_loss, average_precision)


class AssociationTests(unittest.TestCase):
    def test_unknowns_cannot_enter_balanced_sample(self):
        y=np.array([1,0,-1],np.float32);rng=np.random.default_rng(4)
        rows,pairs=sample_rows(y,[],rng,32,4)
        self.assertNotIn(2,rows)
        self.assertEqual(pairs.shape,(0,2))

    def test_competitors_share_mother_and_have_opposite_known_labels(self):
        ix=np.array([[0,2],[0,3],[1,2],[1,3],[4,5]])
        y=np.array([1,0,1,-1,0])
        groups=competing_groups(ix,y)
        self.assertEqual(len(groups),1)
        _,pairs=sample_rows(y,groups,np.random.default_rng(1),8,30)
        self.assertTrue(np.all(ix[pairs[:,0],0]==ix[pairs[:,1],0]))
        self.assertTrue(np.all(y[pairs[:,0]]==1))
        self.assertTrue(np.all(y[pairs[:,1]]==0))

    def test_division_is_invariant_to_daughter_order(self):
        torch.manual_seed(4);model=AssociationRanker(division=True).eval()
        nodes=torch.randn(5,3,68);positions=torch.randn(5,3,3)
        self.assertTrue(torch.allclose(model(nodes,positions),model(nodes[:,[0,2,1]],positions[:,[0,2,1]]),atol=1e-6))

    def test_ranking_gradient_promotes_positive_and_demotes_negative(self):
        pairs=torch.tensor([[0.,0.]],requires_grad=True)
        loss=ranking_loss(torch.zeros(2),torch.tensor([1.,0.]),pairs);loss.backward()
        self.assertLess(pairs.grad[0,0],0)
        self.assertGreater(pairs.grad[0,1],0)

    def test_tied_ap_is_prevalence_not_input_order(self):
        self.assertAlmostEqual(average_precision([0,0,0,0],[1,0,0,0]),.25)
        self.assertAlmostEqual(average_precision([0,0,0,0],[0,0,0,1]),.25)
        self.assertAlmostEqual(average_precision([4,3,2,1],[1,1,0,0]),1.)

    def test_toy_competing_links_are_learnable(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        model=AssociationRanker();optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        nodes=torch.zeros(16,2,68);p=torch.zeros(16,2,3)
        p[8:,1,0]=1.;labels=torch.cat((torch.ones(8),torch.zeros(8)))
        for _ in range(30):
            scores=model(nodes,p);loss=ranking_loss(scores,labels,torch.stack((scores[:8],scores[8:]),1))
            optimizer.zero_grad();loss.backward();optimizer.step()
        scores=model(nodes,p).detach()
        self.assertGreater(float(scores[:8].mean()-scores[8:].mean()),2.)


if __name__=='__main__':unittest.main()
