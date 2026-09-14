import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from biohub_lab.detector_proposals import augment_detections, nms, temporal_filter, patch_predictor, install_pipeline_hook
from biohub_lab.gaussian_detector import model_and_jac, fit_split, GRID
from biohub_lab.cellect_detector import preprocess, tile_starts, decode, find_checkpoint, WEIGHT_NAMES

ROOT=Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def test_checkpoint_accepts_upstream_and_kaggle_names_with_pinned_content(self):
        content=b'pinned checkpoint fixture'
        digest=hashlib.sha256(content).hexdigest()
        for name in WEIGHT_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);asset=root/'datasets'/'jarturo'/'cellect'/name
                asset.parent.mkdir(parents=True);asset.write_bytes(content)
                self.assertEqual(find_checkpoint(root,expected_sha=digest),asset.resolve())
                asset.write_bytes(b'wrong weights')
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                    find_checkpoint(root,expected_sha=digest)

    def test_checkpoint_missing_or_ambiguous_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'exactly one'):
                find_checkpoint(root)
            for name in WEIGHT_NAMES:(root/name).write_bytes(b'same')
            with self.assertRaisesRegex(ValueError,'exactly one'):
                find_checkpoint(root)

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
        # Match the upstream nesting and run the installed hook in a clean scope.
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
        self.assertLess(changed.index('install_predictor_file(_ps,'),changed.index('def list_test_stems'))
        self.assertGreater(changed.index('install_predictor_file(_ps,'),changed.index('_runtime_integrity_receipt_path.write_text'))
        # The baseline does not define hashlib here. The full injected block must
        # run with just the two baseline path variables, not our test imports.
        anchor='print("secondary edge-feature TTA patch installed and enabled", flush=True)'
        block=install_pipeline_hook(anchor)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);predictor=root/'predictor.py';predictor.write_text(source)
            namespace={'_ps':predictor,'WORKING_DIR':root}
            exec(compile(block,'<installed-hook>','exec'),namespace)
            receipt=json.loads((root/'proposal_patch_receipt.json').read_text())
            self.assertEqual(receipt['before_sha256'],hashlib.sha256(source.encode()).hexdigest())
            self.assertEqual(receipt['after_sha256'],hashlib.sha256(patched.encode()).hexdigest())
            self.assertEqual(predictor.read_text(),patched)
            self.assertNotIn('hashlib',namespace)
            np.savez_compressed(root/'sample.npz',coords=np.array([[3,10,60,40]],np.float32),
                                scores=np.array([.9]),shape=np.array([100,64,256,256]))
            execute={}
            exec(compile(predictor.read_text(),str(predictor),'exec'),execute)
            with patch.dict(os.environ,{'BIOHUB_PROPOSAL_DIR':str(root),'BIOHUB_GPU_SHARD':'test'}):
                result,offset=execute['predict'](np.array([[3,10,10,10]]),root/'sample.zarr',3,(1,4,4))
            np.testing.assert_array_equal(result,[[3,10,10,10],[3,10,15,10]])
            self.assertEqual(offset,{3:(0,2)})
            audit=json.loads((root/'injection_test.jsonl').read_text())
            self.assertEqual((audit['baseline'],audit['proposals'],audit['added']),(1,1,1))


if __name__=='__main__':unittest.main()
