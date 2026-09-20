import unittest
import numpy as np
from biohub_lab.nucverse_tiles import predict_volume

class TilesTest(unittest.TestCase):
    def test_overlap_and_reflected_border_reconstruct_same_signal(self):
        image=np.arange(5*7*9,dtype=np.float32).reshape(5,7,9)
        def model(x):return np.concatenate([1-x,x],axis=-1),np.repeat(x,3,axis=-1)
        actual,report=predict_volume(model,image,patch=(8,4,4),stride=(4,3,3))
        low,high=np.percentile(image,[2,99.8]);expected=np.clip((image-low)/(high-low),0,1)
        self.assertGreater(report['tiles'],1)
        for c in range(4):np.testing.assert_allclose(actual[...,c],expected,atol=1e-6)
    def test_nonfinite_predictions_are_rejected(self):
        def model(x):return np.full(x.shape[:-1]+(2,),np.nan),np.zeros(x.shape[:-1]+(3,))
        with self.assertRaises(AssertionError):predict_volume(model,np.ones((2,2,2)),(2,2,2),(2,2,2))
