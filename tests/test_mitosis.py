import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from biohub_lab.dense_lineage import render_scene,build_cache
from biohub_lab.mitosis_model import MitosisSpecialist
from biohub_lab.mitosis_train import ranking_metrics,operating_counts,train_specialist
from biohub_lab.mitosis_repair import repair_candidates,select_repairs,repair_video,load_specialist
from biohub_lab.temporal_data import TemporalConfig,examples
from biohub_lab.temporal_model import TemporalLinker
from biohub_lab.temporal_train import CacheStore,make_split


class MitosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_event_symmetry_temporal_order_and_frozen_representation(self):
        torch.manual_seed(27)
        model=MitosisSpecialist().eval()
        image=torch.randint(0,256,(2,3,5,9,17,17),dtype=torch.uint8)
        delta=torch.randn(2,2,3)
        original=model(image,delta)
        swapped=model(image[:,[0,2,1]],delta[:,[1,0]])
        torch.testing.assert_close(original,swapped,atol=1e-6,rtol=1e-5)
        reversed_time=model(image.flip(2),delta)
        self.assertGreater(float((original-reversed_time).abs().max().detach()),1e-6)
        model.train();self.assertFalse(model.backbone.training)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(image,delta),torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.frame_encoder[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.time_encoder.weight_ih_l0.grad.abs().sum()),0.)

    def test_long_synthetic_divisions_have_six_real_context_frames(self):
        volume,coords,edges,events=render_scene(5,cells=8,division_probability=1.,frames=9)
        self.assertEqual(len(volume),9);self.assertEqual(len(events),8)
        for event in events:
            self.assertGreaterEqual(event['t']-2,0)
            self.assertLess(event['t']+3,len(volume))
        self.assertTrue(np.all(coords[edges[:,1],0]==coords[edges[:,0],0]+1))

    def test_development_threshold_respects_ties_and_abstains_without_classes(self):
        tied=ranking_metrics([1,0],[2.,2.])
        self.assertEqual(tied['average_precision'],.5)
        self.assertEqual((tied['threshold_tp'],tied['threshold_fp']),(1,1))
        scored=ranking_metrics([1,0,1,0],[4.,3.,2.,1.])
        self.assertEqual(scored['threshold'],4.)
        self.assertEqual(operating_counts([1,0],[4.,3.],scored['threshold'])['fp'],0)
        self.assertIsNone(ranking_metrics([0,0],[5.,7.])['threshold'])
        self.assertEqual(operating_counts([1,0],[4.,3.],None)['tp'],0)

    def test_global_orphan_assignment_preserves_base_and_beats_greedy(self):
        coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,10],[1,0,0,5],[1,0,0,12]],np.float32)
        base=np.array([[0,2],[1,3]])
        proposed=repair_candidates(coords,base)
        self.assertEqual(set(map(tuple,proposed)),{(0,2,4),(0,2,5),(1,3,4),(1,3,5)})
        triples=np.array([[0,2,4],[0,2,5],[1,3,4],[1,3,5]])
        repaired,stats,_=select_repairs(coords,base,triples,[8.,7.,6.,1.],0.)
        self.assertEqual(set(map(tuple,repaired)),{(0,2),(1,3),(0,5),(1,4)})
        self.assertEqual(stats['removed_baseline_edges'],0)
        self.assertEqual(stats['added_edges'],2)
        unchanged,_,_=select_repairs(coords,base,triples,[8.,7.,6.,1.],None)
        np.testing.assert_array_equal(unchanged,base)
        with self.assertRaisesRegex(ValueError,'overwrite'):
            select_repairs(coords,base,[[0,2,3]],[10.],0.)
        empty,_,_=select_repairs(coords,base,[],[],0.)
        np.testing.assert_array_equal(empty,base)

    def test_two_stage_training_reload_inference_and_provenance_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=root/'cache';cache.mkdir()
            names=[f'{g}_{i}' for g in ('a','b') for i in range(6)]
            (cache/'cache_manifest.json').write_text(json.dumps(dict(complete=True,videos=[dict(name=n) for n in names])))
            coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,2],[1,0,0,10],[1,0,0,20]],np.float32)
            edges=np.array([[0,2],[0,3],[1,4]],np.int64);rng=np.random.default_rng(7)
            for name in names:
                (cache/name).mkdir()
                np.savez(cache/name/'graph.npz',coords=coords,edges=edges,**examples(coords,edges))
                np.save(cache/name/'patches.npy',rng.integers(0,256,(6,5,9,17,17),dtype=np.uint8))
            store=CacheStore(cache);split=make_split(names,'b')
            build_cache(store,split,root/'synthetic',scenes=4,frames=9);del store
            initial=root/'initial.pt'
            torch.save(dict(state_dict=TemporalLinker().state_dict(),config=asdict(TemporalConfig()),split=split),initial)
            sha=hashlib.sha256(initial.read_bytes()).hexdigest()
            with patch('torch.cuda.is_available',return_value=False):
                result=train_specialist(cache,root/'synthetic',initial,root/'trained','b',
                    pretrain_steps=2,finetune_steps=2,per_class=2,replay_per_class=1,expected_hash=sha)
                with self.assertRaisesRegex(ValueError,'checksum'):
                    train_specialist(cache,root/'synthetic',initial,root/'bad','b',expected_hash='0'*64)
                with self.assertRaisesRegex(ValueError,'split/config'):
                    train_specialist(cache,root/'synthetic',initial,root/'wrong','a',expected_hash=sha)
            self.assertTrue(result['frozen_backbone_unchanged'])
            self.assertEqual(result['holdout']['positive'],6)
            self.assertNotIn('threshold',result['holdout'])
            model,checkpoint=load_specialist(root/'trained/best.pt',torch.device('cpu'))
            volume=rng.integers(0,256,(3,9,17,17),dtype=np.uint8)
            baseline=np.array([[0,2],[1,4]],np.int64)
            with patch('biohub_lab.mitosis_repair.load_volume',return_value=volume):
                repaired,stats,arrays=repair_video(model,checkpoint,coords,baseline,'unseen.zarr',torch.device('cpu'))
            self.assertTrue(set(map(tuple,baseline))<=set(map(tuple,repaired)))
            self.assertEqual(stats['removed_baseline_edges'],0)
            self.assertTrue(np.isfinite(arrays['scores']).all())


if __name__=='__main__': unittest.main()
