import unittest
import numpy as np
import torch
from biohub_lab.temporal_model import TemporalLinker
from biohub_lab.temporal_train import make_split,train_fold
from biohub_lab.temporal_inference import solve_hypotheses
from biohub_lab.temporal_data import TemporalConfig, examples, extract_patches, parent_candidates


class TemporalDataTests(unittest.TestCase):
    def test_context_is_centered_and_boundary_padding_does_not_wrap(self):
        cfg=TemporalConfig(patch_shape=(3,3,3),downsample=(1,1,1))
        vol=np.stack([np.full((4,4,4),t+1,np.uint8) for t in range(7)])
        out=extract_patches(vol,np.array([[3,2,2,2],[0,0,0,0]]),cfg)
        np.testing.assert_array_equal(out[0,:,1,1,1],[2,3,4,5,6])
        np.testing.assert_array_equal(out[1,:,1,1,1],[1,1,1,2,3])
        self.assertTrue((out[1,:,:,0,0]==0).all())

    def test_candidates_use_physical_distance_and_consecutive_time(self):
        cfg=TemporalConfig(max_parents=3,max_distance_um=2.)
        xyz=np.array([[0,0,0,0],[0,2,0,0],[0,0,0,4],[1,0,0,0],[2,0,0,0]])
        parents=parent_candidates(xyz,cfg)
        self.assertEqual(set(parents[3]),{0,2,-1})
        self.assertEqual(set(parents[4]),{3,-1})

    def test_unknown_targets_are_unsupervised_and_pair_negatives_are_known_wrong(self):
        xyz=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,2],[1,0,0,10],[1,0,0,20]])
        ex=examples(xyz,np.array([[0,2],[0,3],[1,4]]))
        self.assertEqual(ex['labels'][5],-1)
        self.assertEqual(int(ex['triple_labels'].sum()),1)
        for (p,a,b),y in zip(ex['triples'],ex['triple_labels']):
            self.assertNotIn(5,(a,b))
            if y==0: self.assertTrue(ex['true_parent'][a]!=p or ex['true_parent'][b]!=p)


class TemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_acquisition_split_excludes_whole_prefix(self):
        names=[f'{g}_{i}' for g in ('a','b') for i in range(10)]
        split=make_split(names,'b')
        self.assertEqual(set(split['holdout']),{f'b_{i}' for i in range(10)})
        self.assertFalse(set(split['train'])&set(split['dev']))
        self.assertEqual(set(split['train']+split['dev']+split['holdout']),set(names))

    def test_division_daughter_exchange_invariant(self):
        model=TemporalLinker().eval(); h=torch.randn(5,64)
        triple=torch.tensor([[0,1,2],[0,2,1]])
        delta=torch.tensor([[[1.,2.,3.],[2.,-1.,0.]],[[2.,-1.,0.],[1.,2.,3.]]])
        scores=model.divisions(h,triple,delta)
        torch.testing.assert_close(scores[0],scores[1])

    def test_padding_invariance_and_all_missing_parent(self):
        model=TemporalLinker().eval(); h=torch.randn(5,64)
        src=torch.tensor([[0,1,2],[0,1,2]]); target=torch.tensor([3,4])
        delta=torch.randn(2,3,3); mask=torch.tensor([[True,True,False],[False,False,False]])
        a=model.parents(h,src,target,delta,mask)
        h[2]*=100.; b=model.parents(h,src,target,delta,mask)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(a[1].argmax().item(),3)

    def test_joint_losses_reach_image_encoder(self):
        model=TemporalLinker(); patches=torch.randint(0,256,(6,5,9,17,17),dtype=torch.uint8)
        h=model.encode(patches)
        logits=model.parents(h,torch.tensor([[0,1,2],[0,1,2]]),torch.tensor([3,4]),torch.randn(2,3,3),torch.ones(2,3,dtype=torch.bool))
        div=model.divisions(h,torch.tensor([[0,3,4],[1,3,5]]),torch.randn(2,2,3))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))+torch.nn.functional.binary_cross_entropy_with_logits(div,torch.tensor([1.,0.]))
        loss.backward()
        grad=model.encoder[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0.)

    def test_cpu_training_checkpoint_and_heldout_pipeline(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'cache'; root.mkdir()
            coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,2],[1,0,0,10],[1,0,0,20]],np.float32)
            edges=np.array([[0,2],[0,3],[1,4]],np.int64)
            names=[f'{g}_{i}' for g in ('a','b') for i in range(6)]
            (root/'cache_manifest.json').write_text(json.dumps({'complete':True,'videos':[{'name':n} for n in names]}))
            rng=np.random.default_rng(17)
            for name in names:
                (root/name).mkdir()
                np.savez(root/name/'graph.npz',coords=coords,edges=edges,**examples(coords,edges))
                np.save(root/name/'patches.npy',rng.integers(0,256,size=(6,5,9,17,17),dtype=np.uint8))
            output=Path(folder)/'model'
            with patch('torch.cuda.is_available',return_value=False): train_fold(root,output,'b',steps=2)
            receipt=json.loads((output/'training_receipt.json').read_text())
            self.assertEqual(receipt['holdout']['summary']['n'],18)
            self.assertEqual(receipt['holdout']['divisions']['positive'],6)
            self.assertEqual(receipt['device'],'cpu')
            checkpoint=torch.load(output/'best.pt',weights_only=False)
            self.assertTrue(all(n.startswith('a_') for n in checkpoint['split']['train']))


class TemporalSolverTests(unittest.TestCase):
    def test_fork_requires_explicit_pair_hypothesis(self):
        coords=np.array([[0,0,0,0],[1,0,0,1],[1,0,0,2]])
        edges=np.array([[0,1],[0,2]])
        a,_=solve_hypotheses(coords,edges,[4.,3.],[],[])
        self.assertEqual(len(a),1)
        b,_=solve_hypotheses(coords,edges,[4.,3.],[[0,1,2]],[1.])
        self.assertEqual(set(map(tuple,b)),{(0,1),(0,2)})

    def test_joint_optimizer_matches_exhaustive_event_search(self):
        from itertools import product
        coords=np.array([[0,0,0,0],[0,0,0,4],[1,0,0,0],[1,0,0,2],[1,0,0,4]])
        edges=np.array([[p,d] for p in (0,1) for d in (2,3,4)])
        triples=np.array([[0,2,3],[1,3,4]])
        rng=np.random.default_rng(41)
        for _ in range(8):
            gains=rng.uniform(-1,4,len(edges)); bonuses=rng.uniform(-2,2,len(triples))
            chosen,_=solve_hypotheses(coords,edges,gains,triples,bonuses)
            options=list(map(tuple,edges))+list(map(tuple,triples))
            weights=list(gains)+[gains[list(map(tuple,edges)).index((p,a))]+gains[list(map(tuple,edges)).index((p,b))]+v for (p,a,b),v in zip(triples,bonuses)]
            best=0.
            for take in product((0,1),repeat=len(options)):
                picked=[o for o,x in zip(options,take) if x]
                parents=[o[0] for o in picked]; daughters=[d for o in picked for d in o[1:]]
                if len(set(parents))<len(parents) or len(set(daughters))<len(daughters): continue
                best=max(best,sum(v*x for v,x in zip(weights,take)))
            actual=sum(gains[list(map(tuple,edges)).index(e)] for e in map(tuple,chosen))
            chosen_set=set(map(tuple,chosen))
            for (p,a,b),v in zip(triples,bonuses):
                if (p,a) in chosen_set and (p,b) in chosen_set: actual+=v
            self.assertAlmostEqual(actual,best,places=5)


if __name__=='__main__': unittest.main()
