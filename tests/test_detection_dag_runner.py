from contextlib import redirect_stdout
from dataclasses import asdict
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import detection_dag_runner as runner
from biohub_lab import cellect_detector,temporal_data
from biohub_lab.detector_proposals import save_proposals,SCALE
from test_detection_dag import simple_graph


class DAGRunnerTests(unittest.TestCase):
    def test_fresh_detection_to_frozen_graph_to_real_annotation_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';comp=root/'comp';output=root/'result'
            (package/'baseline').mkdir(parents=True);(comp/'train').mkdir(parents=True);(comp/'test').mkdir()
            names=['a_1','b_1'];graph,coords,edges,config=simple_graph();shape=tuple(map(int,graph['shape']))
            axes=np.meshgrid(*[np.arange(size)*scale for size,scale in zip(shape[1:],SCALE)],indexing='ij')
            grid=np.stack(axes,axis=-1);frames=[]
            for t in range(shape[0]):
                points=coords[coords[:,0]==t,1:]*SCALE
                frame=sum(np.exp(-np.sum((grid-point)**2,axis=-1)/(2*1.5**2)) for point in points)
                frames.append(np.rint(1000*frame).astype(np.uint16))
            for name in names:
                zarr.open_group(str(comp/'train'/(name+'.zarr')),mode='w').create_array('0',data=np.stack(frames))
                gt=zarr.open_group(str(comp/'train'/(name+'.geff')),mode='w')
                gt.create_array('nodes/ids',data=np.arange(len(coords)))
                gt.create_array('edges/ids',data=edges)
                for i,key in enumerate(('t','z','y','x')):gt.create_array('nodes/props/'+key+'/values',data=coords[:,i])
            manifest=dict(config=asdict(runner.DAGConfig()),gate=asdict(runner.CoverageGate()),per_group=1,excluded=[],development=names,
                training_names_sha256=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest())
            (package/'baseline/detection_dag_dev.json').write_text(json.dumps(manifest))
            for relative in ('src/biohub_lab/detection_dag.py','src/biohub_lab/dag_coverage.py','scripts/detection_dag_runner.py'):
                p=package/relative;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes((ROOT/relative).read_bytes())
            def detector(image_path,path,weights):
                frames=[(coords[coords[:,0]==t,1:],np.ones((coords[:,0]==t).sum())) for t in range(shape[0])]
                return save_proposals(path,frames,shape,dict(method='synthetic detector stub'))
            original_load=temporal_data.load_gt
            def annotation(path):
                self.assertTrue((output/'frozen_inputs.json').is_file())
                self.assertEqual(json.loads((output/'result.json').read_text())['status'],'evaluating_frozen_coverage')
                for name in names:self.assertTrue((output/'videos'/name/'dag/graph.npz').is_file())
                return original_load(path)
            def workers(data,root,names,weights):
                for name in names:runner.generate_video(data,root,name,weights)
            with patch.object(runner,'competition_dir',return_value=comp),patch.object(cellect_detector,'find_checkpoint',return_value='synthetic.pt'),patch.object(cellect_detector,'run_video',side_effect=detector),patch.object(runner,'run_workers',side_effect=workers),patch.object(temporal_data,'load_gt',side_effect=annotation) as gt_reader,redirect_stdout(io.StringIO()):
                result=runner.main(package,output)
            self.assertEqual(result['status'],'complete');self.assertEqual(gt_reader.call_count,2)
            self.assertEqual(result['readiness']['totals']['events'],2)
            self.assertEqual(result['readiness']['totals']['annotated_paths_covered'],2)
            self.assertFalse(result['readiness']['coverage_gate_passed']) # sample is deliberately underpowered
            self.assertFalse(result['neural_training']);self.assertFalse((output/'submission.csv').exists())
            frozen=json.loads((output/'frozen_inputs.json').read_text())
            with (output/'videos/a_1/dag/graph.npz').open('ab') as handle:handle.write(b'changed')
            with patch.object(temporal_data,'load_gt') as gt_reader:
                with self.assertRaisesRegex(ValueError,'changed before'):runner.audit_frozen(output,comp/'train',frozen)
                gt_reader.assert_not_called()


if __name__=='__main__':unittest.main()
