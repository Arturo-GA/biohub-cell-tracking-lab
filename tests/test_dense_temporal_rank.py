import unittest
import numpy as np
from dense_temporal_rank_runner import examples

class TemporalExamplesTests(unittest.TestCase):
    def test_duplicate_hypothesis_remains_unknown(self):
        coords=np.array([[0,0,0,0],[0,0,0,8],[0,0,0,25],[1,0,0,1]],float)
        truth=np.array([[10,0,0,0,0],[20,1,0,0,1]],float)
        x,y,m,c=examples(coords,np.ones((4,64),np.float32),truth,[[10,20]])
        self.assertEqual(x.shape,(1,12,132))
        self.assertEqual(int(y[0]),0)
        self.assertTrue(m[0,0,1]);self.assertFalse(m[0,1,1])
        self.assertTrue(m[0,1,2]);self.assertEqual(c['represented'],1)

    def test_missing_true_parent_is_not_fabricated(self):
        coords=np.array([[0,0,0,30],[0,0,0,35],[1,0,0,1]],float)
        truth=np.array([[10,0,0,0,0],[20,1,0,0,1]],float)
        x,y,m,c=examples(coords,np.ones((3,64),np.float32),truth,[[10,20]])
        self.assertEqual(len(y),0);self.assertEqual(c['represented'],0)

if __name__=='__main__':unittest.main()
