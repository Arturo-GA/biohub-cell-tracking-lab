import unittest,numpy as np
from biohub_lab.public_divnet import input_patch
class PublicDivnetTests(unittest.TestCase):
    def test_marker_preserves_fractional_center(self):
        p=input_patch(np.ones((3,20,40,40),np.float32),[1,8,65,64])
        self.assertEqual(p.shape,(5,16,32,32));self.assertTrue(np.all(np.isfinite(p)))
        self.assertEqual(np.unravel_index(p[4].argmax(),p[4].shape),(8,16,16))
        self.assertGreater(p[4,8,17,16],p[4,8,15,16])
    def test_time_boundary_and_reflection(self):
        v=np.arange(4*20*40*40,dtype=np.float32).reshape(4,20,40,40)
        p=input_patch(v,[0,0,0,0]);np.testing.assert_array_equal(p[0],p[1]);self.assertTrue(np.all(np.isfinite(p)))
if __name__=='__main__':unittest.main()
