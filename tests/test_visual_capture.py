import tempfile,unittest
from pathlib import Path
import numpy as np
from biohub_lab.visual_capture import capture_pair,patch_predictor,inject_baseline

class CaptureTests(unittest.TestCase):
    def test_capture_does_not_mutate_and_keeps_competing_maxima(self):
        rng=np.random.default_rng(22);p=rng.random((12,15));before=p.copy()
        a=np.zeros((12,4),np.int16);b=np.zeros((15,4),np.int16);b[:,0]=1;a[:,1:]=2;b[:,1:]=3
        with tempfile.TemporaryDirectory() as tmp:
            capture_pair('sample',0,a,b,np.arange(12),np.arange(15),p,[1,4,4],tmp)
            with np.load(Path(tmp)/'sample/0.npz') as f:
                edges=set(map(tuple,f['edges']))
                self.assertTrue(all((i,int(p[i].argmax())) in edges for i in range(12)))
                self.assertTrue(all((int(p[:,j].argmax()),j) in edges for j in range(15)))
                np.testing.assert_array_equal(f['source_coords'][:,1:],np.tile([2,8,8],(12,1)))
        np.testing.assert_array_equal(p,before);np.testing.assert_array_equal(a[:,1:],2)
    def test_frozen_runtime_patches_compile(self):
        root=Path(__file__).resolve().parents[1]
        source='def predict():\n    for frame in frames:\n        if frame:\n            candidates = sorted(\n                []\n            )\n'
        patched=patch_predictor(source)
        self.assertEqual(patched.count('capture_pair(ds_path.stem'),1)
        inject_baseline((root/'baseline/harmonic_inference.py').read_text())
        with self.assertRaises(AssertionError):patch_predictor(patched.replace('            candidates = sorted(','            candidates = list('))
if __name__=='__main__':unittest.main()
