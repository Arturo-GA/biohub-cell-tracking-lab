import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from biohub_lab.dense_lineage import render_scene,dense_examples,template_bank,build_cache
from biohub_lab.dense_pretrain import pretrain
from biohub_lab.temporal_train import CacheStore,make_split,train_fold
from biohub_lab.temporal_data import examples


class DenseLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_rendered_divisions_have_complete_binary_graph_and_consecutive_edges(self):
        v,c,e,events=render_scene(17,cells=8,division_probability=1.)
        self.assertEqual(v.dtype,np.uint8);self.assertEqual(len(events),8)
        self.assertTrue(np.all(c[e[:,1],0]==c[e[:,0],0]+1))
        _,incoming=np.unique(e[:,1],return_counts=True);self.assertTrue(np.all(incoming==1))
        _,outgoing=np.unique(e[:,0],return_counts=True);self.assertEqual(int((outgoing==2).sum()),8)
        self.assertTrue((c[:,1:]>=0).all())
        self.assertTrue((c[:,1:]<np.array(v.shape[1:])*[1,4,4]).all())
        ex=dense_examples(c,e);self.assertEqual(int(ex['triple_labels'].sum()),8)
        for (p,a,b),label in zip(ex['triples'],ex['triple_labels']):
            self.assertEqual(bool(label),bool(ex['true_parent'][a]==p and ex['true_parent'][b]==p))

    def test_crossing_is_not_labelled_as_division_and_rendering_is_reproducible(self):
        a=render_scene(5,cells=2,division_probability=0.,force_crossing=True)
        b=render_scene(5,cells=2,division_probability=0.,force_crossing=True)
        for x,y in zip(a[:3],b[:3]): np.testing.assert_array_equal(x,y)
        self.assertEqual(len(a[3]),0)
        ex=dense_examples(a[1],a[2]);self.assertEqual(int(ex['triple_labels'].sum()),0)
        self.assertGreater(len(ex['triples']),0)

    def test_template_guard_rejects_reserved_embryo_before_access(self):
        with self.assertRaisesRegex(ValueError,'Held-out embryo'):
            template_bank(None,['reserved_1'],'reserved')

    def test_pretrain_transfer_end_to_end_and_provenance_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=root/'real';cache.mkdir()
            names=[f'{g}_{i}' for g in ('a','b') for i in range(6)]
            (cache/'cache_manifest.json').write_text(json.dumps(dict(complete=True,videos=[dict(name=n) for n in names])))
            coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,2],[1,0,0,10],[1,0,0,20]],np.float32)
            edges=np.array([[0,2],[0,3],[1,4]],np.int64);rng=np.random.default_rng(7)
            for n in names:
                (cache/n).mkdir()
                np.savez(cache/n/'graph.npz',coords=coords,edges=edges,**examples(coords,edges))
                np.save(cache/n/'patches.npy',rng.integers(0,256,(6,5,9,17,17),dtype=np.uint8))
            store=CacheStore(cache);split=make_split(names,'b')
            manifest=build_cache(store,split,root/'synthetic',scenes=4)
            del store  # Release Windows mmap handles before temporary-directory cleanup.
            self.assertGreater(manifest['generated_divisions'],0)
            self.assertTrue(all(x['video'] in split['train'] for x in manifest['template_sources']))
            with patch('torch.cuda.is_available',return_value=False):
                initial=pretrain(root/'synthetic',root/'pretrain',steps=2)
                train_fold(cache,root/'adapted','b',steps=2,initial_checkpoint=initial)
                with self.assertRaisesRegex(ValueError,'split/config mismatch'):
                    train_fold(cache,root/'bad','a',steps=1,initial_checkpoint=initial)
            model=torch.load(root/'adapted/best.pt',weights_only=False)
            self.assertIn('Dense procedural',model['initialization'])
            self.assertEqual(len(model['initial_checkpoint_sha256']),64)
            self.assertEqual(model['split']['holdout'],split['holdout'])


if __name__=='__main__': unittest.main()
