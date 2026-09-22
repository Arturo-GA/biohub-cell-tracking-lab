import unittest
import numpy as np
from biohub_lab.trajectory_denoise import smooth_path,refine,ARMS

class DenoiseTests(unittest.TestCase):
    def test_linear_motion_is_fixed_point(self):
        p=np.arange(24,dtype=float).reshape(8,3);self.assertTrue(np.allclose(smooth_path(p,8.),p,atol=1e-10))
    def test_reduces_isolated_jitter(self):
        p=np.zeros((9,3));p[:,0]=np.arange(9);p[4,1]=4.;q=smooth_path(p,2.);self.assertLess(abs(q[4,1]),2.);self.assertTrue(np.isfinite(q).all())
    def test_does_not_smooth_across_divisions(self):
        n={i:dict(t=i,z=2,y=5,x=5+i) for i in range(3)};n[3]=dict(t=2,z=2,y=5,x=1);e=[(0,1),(1,2),(1,3)];new,r=refine(n,e,**ARMS['strong'],shape=(4,5,20,20));self.assertEqual(new,n);self.assertEqual(r['smoothed_paths'],0)
    def test_bounds_and_nonempty_movement(self):
        n={i:dict(t=i,z=2,y=5+(4 if i==4 else 0),x=5+i) for i in range(9)};e=list(zip(range(8),range(1,9)));new,r=refine(n,e,**ARMS['medium'],shape=(9,5,20,20));self.assertGreater(r['moved'],0);self.assertEqual(set(n),set(new));self.assertLessEqual(r['max_shift_um'],2.)

if __name__=='__main__':unittest.main()
