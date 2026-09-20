import unittest
import numpy as np
import torch
from biohub_lab.temporal_volume import records,crop_temporal,TemporalVolumeEncoder,logits

class TemporalVolumeTests(unittest.TestCase):
    def test_time_channels_and_border(self):
        image=np.stack([np.full((4,4,4),i) for i in range(3)])
        p=np.pad(image,((0,0),(8,8),(8,8),(8,8)),mode='edge')
        c=crop_temporal(p,np.array([0,0,0,0]));self.assertEqual(c.shape,(3,16,16,16))
        np.testing.assert_array_equal(c[:,8,8,8],[0,0,1])

    def test_known_close_neighbor_is_negative_unknown_is_masked(self):
        coords=np.array([[0,0,0,0],[0,0,0,8],[0,0,0,10],[1,0,0,1]],float)
        truth=np.array([[10,0,0,0,0],[11,0,0,0,8],[20,1,0,0,1]],float)
        d=records(coords,truth,[[10,20]],{10:0,11:1,20:3})
        self.assertTrue(d['mask'][0,1]);self.assertFalse(d['mask'][0,2]);self.assertTrue(d['valid'][0,2])

    def test_convolution_receives_temporal_learning_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(33);model=TemporalVolumeEncoder()
        before=model.net[0].weight.detach().clone();opt=torch.optim.Adam(model.parameters(),lr=.001)
        h=model(torch.randn(6,3,16,16,16)).reshape(2,3,64)
        score,appearance=logits(h,torch.tensor([[2.,3.],[3.,2.]]))
        loss=torch.nn.functional.cross_entropy(score,torch.tensor([0,1]))+.5*torch.nn.functional.cross_entropy(appearance,torch.tensor([0,1]))
        loss.backward();self.assertGreater(float(model.net[0].weight.grad.abs().sum()),0)
        opt.step();self.assertFalse(torch.equal(before,model.net[0].weight))

if __name__=='__main__':unittest.main()
