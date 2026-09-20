import unittest
from collections import Counter
import numpy as np
import torch
from biohub_lab.temporal_graph import reassign
from biohub_lab.temporal_volume import crop_temporal,gather_temporal

class TemporalGraphTests(unittest.TestCase):
    def test_image_can_swap_parents_without_changing_degrees(self):
        ids=np.arange(4);coords=np.array([[0,0,0,0],[0,0,0,8],[1,0,0,0],[1,0,0,8]],float)
        edges=[(0,2),(1,3)];features=np.array([[1,0],[0,1],[0,1],[1,0]],float)
        geo,_=reassign(ids,coords,edges,features,False);learned,_=reassign(ids,coords,edges,features,True)
        self.assertEqual(geo,edges);self.assertEqual(learned,[(0,3),(1,2)])
        self.assertEqual(Counter(a for a,b in learned),Counter(a for a,b in edges))

    def test_divisions_and_gaps_are_locked(self):
        ids=np.arange(5);coords=np.array([[0,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,2],[2,0,0,2]],float)
        edges=[(0,1),(0,2),(3,4)];h=np.ones((5,2))
        result,report=reassign(ids,coords,edges,h);self.assertEqual(result,edges);self.assertEqual(report['locked_edges'],3)

    def test_vectorized_crops_equal_training_crops(self):
        rng=np.random.default_rng(33);image=rng.normal(size=(3,4,4,4)).astype(np.float16)
        padded=np.pad(image,((0,0),(8,8),(8,8),(8,8)),mode='edge')
        coords=np.array([[0,0,0,0],[2,3,12,12],[1,2,6,10]],float)
        expected=np.stack([crop_temporal(padded,c) for c in coords])
        centers=torch.from_numpy(np.rint(coords/np.array([1,1,4,4])).astype(np.int64))
        actual=gather_temporal(torch.from_numpy(padded),centers).numpy()
        np.testing.assert_array_equal(actual,expected)

if __name__=='__main__':unittest.main()
