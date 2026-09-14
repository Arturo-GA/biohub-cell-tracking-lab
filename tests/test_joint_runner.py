"""Exercise the actual cache adapter, image loader, reconstruction and CSV writer."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import joint_runner as runner
from test_joint_lineage import scene


def fixture(folder):
    folder=Path(folder);package=folder/'package';inputs=folder/'inputs';data=folder/'data'
    (package/'baseline').mkdir(parents=True);data.mkdir()
    for relative in ('src/biohub_lab/joint_lineage.py','scripts/joint_runner.py'):
        target=package/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    base,edges,proposals,shape,evidence=scene()
    # Actual Zarr path and raw-intensity preprocessing; no annotation file exists.
    raw=(1000*evidence.volume.repeat(2,axis=2).repeat(2,axis=3)).astype(np.uint16)
    zarr.open_group(str(data/'synthetic.zarr'),mode='w').create_array('0',data=raw,chunks=(1,*shape[1:]))
    control=inputs/'control';(control/'biohub_control').mkdir(parents=True)
    csv=control/'biohub_control/submission.csv';runner.write_csv(csv,{'synthetic':(base,edges)})
    summary=dict(score=0.,adj_edge_jaccard=0.,division_jaccard=0.)
    receipt=dict(datasets=['synthetic'],arms={'control':dict(sha256=runner.sha256(csv),metrics={'summary':summary})})
    (control/'run_receipt.json').write_text(json.dumps(receipt))
    manifest=dict(datasets=['synthetic'],control=dict(slug='control',receipt_sha256=runner.sha256(control/'run_receipt.json'),csv_sha256=runner.sha256(csv)),proposals=[])
    for method in ('cellect','gaussian'):
        root=inputs/method/'detector_experiment';(root/'proposals').mkdir(parents=True)
        path=root/'proposals/synthetic.npz'
        np.savez_compressed(path,coords=proposals,scores=np.ones(len(proposals)),shape=shape)
        record=dict(sha256=runner.sha256(path),proposals=len(proposals))
        (root/'result.json').write_text(json.dumps(dict(status='complete',method=method,datasets=['synthetic'],proposal_receipts={'synthetic':record})))
        manifest['proposals'].append(dict(slug=method,method=method,receipt_sha256=runner.sha256(root/'result.json'),files={'synthetic':dict(sha256=record['sha256'],count=len(proposals))}))
    (package/'baseline/joint_inputs.json').write_text(json.dumps(manifest))
    return package,inputs,data,manifest


class JointRunnerTests(unittest.TestCase):
    def test_pipeline_freezes_valid_csv_before_evaluator_and_preserves_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            package,inputs,data,manifest=fixture(folder);output=Path(folder)/'output'
            def evaluate(csv,data_dir):
                result=json.loads((output/'result.json').read_text())
                self.assertEqual(result['status'],'evaluating')
                self.assertEqual(result['csv_sha256'],runner.sha256(csv))
                self.assertTrue((output/'synthetic/event_audit.jsonl').exists())
                self.assertFalse(list(data_dir.glob('*.geff')))
                runner.read_and_validate(csv,runner.shapes_for(data_dir))
                return dict(summary=dict(score=0.,adj_edge_jaccard=0.,division_jaccard=0.),samples=[])
            with patch.object(runner,'competition_dir',return_value=data),patch.object(runner,'diagnostic_data',return_value=(data,['synthetic'])),patch.object(runner,'evaluate_csv',side_effect=evaluate) as metric,redirect_stdout(io.StringIO()):
                result=runner.main(package,output,inputs)
            self.assertEqual(result['status'],'complete');metric.assert_called_once()
            self.assertEqual(result['selected_events'],1)
            self.assertTrue(result['validated'])
            self.assertFalse((output.parent/'submission.csv').exists())

    def test_changed_cache_fails_before_reconstruction_and_writes_error(self):
        with tempfile.TemporaryDirectory() as folder:
            package,inputs,_,_=fixture(folder);output=Path(folder)/'output'
            path=inputs/'cellect/detector_experiment/proposals/synthetic.npz'
            with path.open('ab') as handle:handle.write(b'changed')
            with patch.object(runner,'reconstruct') as reconstruct:
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):runner.main(package,output,inputs)
                reconstruct.assert_not_called()
            receipt=json.loads((output/'result.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            self.assertEqual(receipt['failed_stage'],'verifying_inputs')
            self.assertTrue((output/'error.txt').exists())

    def test_sparse_original_ids_remap_edges_correctly(self):
        nodes={99:dict(t=3,z=1,y=2,x=3),20:dict(t=2,z=1,y=2,x=2)}
        coords,edges=runner.graph_arrays(nodes,[(20,99)])
        np.testing.assert_array_equal(coords[:,0],[2,3])
        np.testing.assert_array_equal(edges,[[0,1]])


if __name__=='__main__':unittest.main()
