import ast
from pathlib import Path
import unittest
import numpy as np
from biohub_lab.detector_proposals import augment_detections, nms, temporal_filter, patch_predictor, install_pipeline_hook
from biohub_lab.gaussian_detector import model_and_jac, fit_split, GRID
from biohub_lab.cellect_detector import preprocess, tile_starts, decode

ROOT=Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def test_cellect_packaged_source_integrity(self):
        import hashlib
        from biohub_lab.cellect_detector import MODEL_SHA
        source=(ROOT/'src/biohub_cellect/model.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(),MODEL_SHA)

    def test_physical_nms_and_native_grid(self):
        base=np.array([[3,10,10,10]],np.int16)  # native 10,40,40
        points=np.array([[10,41,40],[10,60,40],[10,61,40]],np.float32)
        result=augment_detections(base,points,np.array([1,.9,.8]),3,(1,4,4),(100,64,256,256))
        np.testing.assert_array_equal(result[0],base[0])
        np.testing.assert_array_equal(result[1],[3,10,15,10])
        self.assertEqual(result.shape,(2,4))
        self.assertEqual(len(nms([[0,0,0],[2,0,0]],[1,1])),2) # 3.25 um
        self.assertEqual(len(nms([[0,0,0],[0,2,0]],[1,1])),1)

    def test_temporal_support_boundaries_and_empty(self):
        frames=[(np.array([[10.,10,10],[40,40,40]]),np.ones(2)),
                (np.array([[10.,10,11]]),np.ones(1)),(np.empty((0,3)),np.empty(0))]
        result=temporal_filter(frames)
        self.assertEqual([len(p) for p,_ in result],[1,1,0])

    def test_gaussian_jacobian(self):
        rng=np.random.default_rng(5); xyz=rng.normal(size=(30,3))
        params=np.array([.03,1.2,1.4,1.5,.6,0,0,-2,.4,0,0,2])
        _,jac=model_and_jac(params,xyz,2)
        for i in range(len(params)):
            direction=np.zeros(len(params));direction[i]=1.e-6
            numerical=(model_and_jac(params+direction,xyz,2)[0]-model_and_jac(params-direction,xyz,2)[0])/2.e-6
            np.testing.assert_allclose(jac[:,i],numerical,atol=1.e-7)

    def test_separates_overlap_and_keeps_single_elongated_nucleus(self):
        shape=(9,19,19)
        p=np.stack(np.meshgrid(*[(np.arange(n)-(n-1)/2)*s for n,s in zip(shape,GRID)],indexing='ij'),-1)
        noise=np.random.default_rng(42).normal(0,.003,shape)+.03
        # A single elongated Gaussian must not become two detections.
        single=.7*np.exp(-.5*np.sum((p/[1.5,1.5,2.5])**2,-1))+noise
        self.assertIsNone(fit_split(single))
        pair=sum(.6*np.exp(-.5*np.sum(((p-[0,0,c])/1.5)**2,-1)) for c in (-2.,2.))+noise
        result=fit_split(pair)
        self.assertIsNotNone(result)
        self.assertLess(np.max(np.abs(np.sort(result[0][:,2])-[-2,2])),.1)

    def test_cellect_normalization_and_tile_coverage(self):
        self.assertTrue(np.isfinite(preprocess(np.zeros((2,3,4,5)))).all())
        starts=tile_starts(64,32,8)
        covered=set(x for start in starts for x in range(start+1,start+31))
        self.assertTrue(set(range(1,63))<=covered)

    def test_cellect_peak_axes_and_foreground(self):
        import torch
        seg=torch.zeros(1,3,24,24,8);seg[:,1]=10
        loc=torch.zeros(1,5,24,24,8);loc[:,0]=1
        loc[0,4,12,14,4]=100
        coords,scores=decode(seg,loc)
        self.assertTrue(any(np.array_equal(p,[12,14,4]) for p in coords))
        seg[:,0]=20
        self.assertEqual(len(decode(seg,loc)[0]),0)

    def test_runtime_hook_before_registration_and_fail_closed(self):
        # Execute a minimal upstream-shaped predictor to verify call ordering.
        source='''def predict(arr, ds_path, t, downsample):
    coord_offset={}
    global_node_count=0
    if True:
        if True:
            if True:
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
    return arr,coord_offset
'''
        patched=patch_predictor(source)
        self.assertLess(patched.index('arr = augment_runtime'),patched.index('coord_offset[t]'))
        with self.assertRaises(ValueError):patch_predictor(patched)
        with self.assertRaises(ValueError):patch_predictor('x=1')
        baseline=(ROOT/'baseline/harmonic_inference.py').read_text()
        changed=install_pipeline_hook(baseline)
        ast.parse(changed)
        self.assertLess(changed.index('_proposal_changed = patch_predictor'),changed.index('def list_test_stems'))
        self.assertGreater(changed.index('_proposal_changed = patch_predictor'),changed.index('_runtime_integrity_receipt_path.write_text'))


if __name__=='__main__':unittest.main()
