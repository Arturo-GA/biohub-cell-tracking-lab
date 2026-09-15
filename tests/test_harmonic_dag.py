from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr
from biohub_lab.harmonic_centers import image_metadata,merge_primary,detector_source
from biohub_lab.detection_dag import build_dag,save_dag
from biohub_lab import temporal_data
from test_detection_dag import simple_graph

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import harmonic_dag_runner as runner


class HarmonicDAGTests(unittest.TestCase):
    def test_primary_retained_regardless_of_auxiliary_score(self):
        primary=np.array([[0,4,12,12],[0,4,13,12]],np.int32)
        aux=(np.array([[0,4,12,13],[0,4,30,30]],np.float32),np.array([1.e6,.2],np.float32))
        coords,scores,origin=merge_primary(primary,[aux],(2,8,64,64))
        np.testing.assert_array_equal(coords,[[0,4,12,12],[0,4,13,12],[0,4,30,30]])
        np.testing.assert_array_equal(origin,[0,0,1])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            merge_primary(np.concatenate([primary,primary[:1]]),[aux],(2,8,64,64))

    def test_zyx_scale_and_downsampled_shape_without_geff_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.zarr';g=zarr.open_group(str(path),mode='w')
            g.create_array('0',data=np.zeros((3,9,25,29),np.uint16))
            g.attrs['image_statistics']={'quantiles':{'0.001':1.,'0.999':99.}}
            # A sibling annotation is deliberately invalid and must not be read.
            path.with_suffix('.geff').write_text('not an annotation database')
            meta=image_metadata(path)
            self.assertEqual(meta.image_shape,(3,9,7,8))
            self.assertEqual(meta.scale,(1.625,.40625,.40625))
            self.assertEqual(meta.quantiles['0.999'],99.)
            with self.assertRaises(ValueError):image_metadata(path.with_suffix('.geff'))

    def test_source_extraction_refuses_tracking_hook_and_missing_boundary(self):
        source='''class PredictConfig: pass
_DEFAULT_CONFIG={}
def load_model(): pass
def _load_frame(): pass
def pool_kernel_from_um(): pass
def _detect_cells_pooled(): pass
def predict_video():
    for ws in []:
        _detect_cells_pooled()
        coords_so_far=[]
        for f_idx in range(W - 1):
            model.predict_edges()
    return [],[]
'''
        trainer='class UNetNodeTransformer: pass'
        changed=detector_source(source,trainer)
        self.assertNotIn('model.predict_edges()',changed)
        self.assertIn('image_metadata as open_dataset',changed)
        with self.assertRaisesRegex(ValueError,'boundary'):
            detector_source(source.replace('range(W - 1)','range(W)'),trainer)
        with self.assertRaisesRegex(ValueError,'Tracking'):
            detector_source(source.replace('coords_so_far=[]','coords_so_far=[]\n        augment_runtime()'),trainer)

    def test_three_arm_audit_is_after_freeze_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'out';cached=root/'cached';data=root/'data'
            out.mkdir();data.mkdir();names=['a_1','b_1'];graph,coords,edges,_=simple_graph();shape=graph['shape']
            pins={'files':{}}
            for name in names:
                p=cached/'videos'/name;p.mkdir(parents=True)
                # E011-like pool has only the mother; adding the primary can restore both branches.
                for method in ('cellect','gaussian'):
                    np.savez_compressed(p/(method+'.npz'),coords=coords[:1],scores=np.ones(1),shape=shape)
                    (p/(method+'.json')).write_text('{}')
                base=build_dag(coords[:1],np.ones(1),np.ones(1),shape);save_dag(base,p/'dag',name)
                pins['files'][name]={f:runner.sha(p/f) for f in ('cellect.npz','cellect.json','gaussian.npz','gaussian.json','dag/graph.npz')}
                runner.generate_video(name,coords,shape,out,cached,{})
                gt=zarr.open_group(str(data/(name+'.geff')),mode='w')
                gt.create_array('nodes/ids',data=np.arange(len(coords)));gt.create_array('edges/ids',data=edges)
                for i,k in enumerate(('t','z','y','x')):gt.create_array('nodes/props/'+k+'/values',data=coords[:,i])
            frozen=runner.freeze_graphs(out,names,cached,pins)
            original=temporal_data.load_gt
            def read_gt(path):
                self.assertTrue((out/'frozen_inputs.json').is_file())
                self.assertTrue(all((out/'videos'/n/'combined/graph.npz').is_file() for n in names))
                return original(path)
            with patch.object(temporal_data,'load_gt',side_effect=read_gt),redirect_stdout(io.StringIO()):
                result=runner.audit_frozen(out,data,cached,frozen)
            self.assertEqual(result['e011']['totals']['centers_covered'],0)
            self.assertEqual(result['combined']['totals']['annotated_paths_covered'],2)
            self.assertEqual(result['harmonic']['totals'],result['combined']['totals'])
            self.assertFalse(result['combined']['coverage_gate_passed']) # synthetic sample is too small
            with (cached/'videos/a_1/dag/graph.npz').open('ab') as handle:handle.write(b'changed')
            with patch.object(temporal_data,'load_gt') as reader:
                with self.assertRaisesRegex(ValueError,'before annotation'):
                    runner.audit_frozen(out,data,cached,frozen)
                reader.assert_not_called()

    def test_cached_input_version_mismatch_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';(package/'baseline').mkdir(parents=True)
            cached=root/'inputs/example/detection_dag_experiment';cached.mkdir(parents=True)
            f=cached/'frozen_inputs.json';f.write_text('{}')
            spec=dict(slug='example',frozen_sha256=hashlib.sha256(b'old').hexdigest())
            (package/'baseline/harmonic_dag_inputs.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'version changed'):runner.verify_cached(package,root/'inputs')


if __name__=='__main__':unittest.main()
