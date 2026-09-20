import unittest
import numpy as np
from biohub_lab.neural_division import event_features,development_threshold
class NeuralDivisionTests(unittest.TestCase):
    def test_symmetric_daughters_and_time_information(self):
        h=np.random.default_rng(3).normal(size=(9,64));c=np.arange(9).reshape(1,9)
        for seq,width in [(False,192),(True,576)]:
            f=event_features(h,c,seq);self.assertEqual(f.shape,(1,width))
            np.testing.assert_allclose(f,event_features(h,c[:,[0,1,2,6,7,8,3,4,5]],seq))
        changed=h.copy();changed[0]+=1
        np.testing.assert_array_equal(event_features(h,c,False),event_features(changed,c,False))
        self.assertFalse(np.array_equal(event_features(h,c,True),event_features(changed,c,True)))
    def test_threshold_no_tie_shortcut_or_single_positive(self):
        self.assertEqual(development_threshold([1,1,0],[.9,.8,.7]),.8)
        self.assertIsNone(development_threshold([1,1,0],[.9,.8,.8]))
        self.assertIsNone(development_threshold([1,0],[.9,.1]))
    def test_empty_events(self):
        self.assertEqual(event_features(np.empty((0,64)),np.empty((0,9)),True).shape,(0,576))
if __name__=='__main__':unittest.main()
