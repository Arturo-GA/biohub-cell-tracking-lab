import collections
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'kaggle/x138_xr'
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
rules=load('independent_rules')

def node(t,x): return dict(t=t,z=0.,y=0.,x=x/.40625)
def edge(a,b,p=.9): return dict(source_id=a,target_id=b,edge_prob=p)

class IndependentTest(unittest.TestCase):
    def crossing(self):
        ns={0:node(0,0),1:node(0,6),2:node(1,2),3:node(1,4),4:node(2,4),5:node(2,2),6:node(3,6),7:node(3,0)}
        es=[edge(a,b) for a,b in [(0,2),(1,3),(2,5),(3,4),(4,6),(5,7)]]
        return ns,es

    def test_geometry_and_neural_use_same_candidates_and_preserve_degrees(self):
        ns,es=self.crossing();cs=rules.swap_candidates(ns,es)
        self.assertEqual(len(cs),1)
        self.assertEqual(rules.apply_swaps(ns,es,cs,mode='geometry')[2]['swaps'],1)
        self.assertEqual(rules.apply_swaps(ns,es,cs)[2]['swaps'],0)
        cs[0].update(new_min_logit=3.,mutual_new=True,min_row_gain=4.,min_col_gain=3.)
        _,result,stats=rules.apply_swaps(ns,es,cs)
        self.assertEqual(stats['swaps'],1)
        for key in ('source_id','target_id'):
            self.assertEqual(collections.Counter(e[key] for e in result),collections.Counter(e[key] for e in es))
        cs[0]['min_col_gain']=-1
        self.assertEqual(rules.apply_swaps(ns,es,cs)[2]['swaps'],0)

    def test_divisions_and_missing_context_protected(self):
        ns,es=self.crossing();ns[8]=node(2,3);es.append(edge(2,8))
        self.assertEqual(rules.swap_candidates(ns,es),[])
        ns,es=self.crossing();ns[6]['t']=4
        self.assertEqual(rules.swap_candidates(ns,es),[])

    def tracks(self,offset=1.):
        ns={i:node(i,i*.2) for i in range(20)}
        ns.update({100+i:node(i,i*.2+offset) for i in range(12)})
        es=[edge(i,i+1,.95) for i in range(19)]+[edge(100+i,101+i,.5) for i in range(11)]
        return ns,es

    def test_duplicates_remove_whole_weak_track_only(self):
        ns,es=self.tracks();out,ee,st=rules.persistent_duplicates(ns,es)
        self.assertEqual(st['removed_tracks'],1);self.assertEqual(st['removed_nodes'],12)
        self.assertEqual(set(out),set(range(20)));self.assertEqual(len(ee),19)
        self.assertEqual(len(ns),32);self.assertEqual(len(es),30)

    def test_close_cells_without_confidence_or_motion_evidence_kept(self):
        ns,es=self.tracks()
        for e in es: e['edge_prob']=.9
        self.assertEqual(rules.persistent_duplicates(ns,es)[2]['removed_tracks'],0)
        ns,es=self.tracks(4.)
        self.assertEqual(rules.persistent_duplicates(ns,es)[2]['removed_tracks'],0)
        ns,es=self.tracks()
        for i in range(12): ns[100+i]['x']+=(1 if i%2 else -1)/.40625
        self.assertEqual(rules.persistent_duplicates(ns,es)[2]['removed_tracks'],0)

    def test_duplicate_track_with_division_is_protected(self):
        ns,es=self.tracks();ns[200]=node(6,4);es.append(edge(5,200))
        self.assertEqual(rules.persistent_duplicates(ns,es)[2]['removed_tracks'],0)
        self.assertEqual(rules.persistent_duplicates({},[])[2]['removed_tracks'],0)

    def test_candidate_annotation_indexing(self):
        linker=load('independent_linker')
        cs=[dict(a=9,b=4,da=20,db=30)]
        out=linker.annotate_candidates(cs,np.array([[7.,-2.],[-3.,8.]]),[4,9],[20,30])[0]
        self.assertTrue(out['mutual_new'])
        self.assertEqual(out['old_logits'],[-3.,-2.]);self.assertEqual(out['new_logits'],[8.,7.])
        self.assertEqual(out['min_col_gain'],10.)

    def test_feature_sampler_uses_native_to_donor_coordinates(self):
        import torch
        linker=load('independent_linker');definitions=load('hengck_model_v12')
        class FakeUNet:
            def make_feature(self,v):return [v,v,v,v,v],v
        class FakeModel:unet=FakeUNet()
        volume=np.indices((8,8,8)).astype(np.float32)
        volume=100*volume[0]+10*volume[1]+volume[2]
        f=linker.sample_frame(FakeModel(),definitions.sample_pyr_feature_at_zyx,volume,
                             [7],{7:dict(z=2,y=12,x=16)},device='cpu')
        self.assertTrue(torch.allclose(f['zyx'],torch.tensor([[[2.,3.,4.]]])))
        self.assertAlmostEqual(float(f['feature'][0][0,0,0]),234.,places=4)

    def test_fragment_join_requires_motion_and_neural_evidence(self):
        ns={k:node(k,k) for k in range(6)}
        es=[edge(0,1),edge(1,2),edge(3,4),edge(4,5)]
        cs=rules.join_candidates(ns,es);self.assertEqual(len(cs),1)
        self.assertEqual(rules.apply_joins(ns,es,cs)[2]['joins'],0)
        cs[0].update(mutual=True,row_margin=3.,col_margin=3.)
        nn,ee,st=rules.apply_joins(ns,es,cs)
        self.assertEqual(st['joins'],1);self.assertEqual(nn,ns)
        self.assertIn((2,3),[(e['source_id'],e['target_id']) for e in ee])
        self.assertEqual(rules.apply_joins(ns,ee,cs)[2]['joins'],0)
        ns[6]=node(2,3);es.append(edge(1,6))
        self.assertEqual(rules.join_candidates(ns,es),[])

    def test_join_margin_includes_null_and_competing_cells(self):
        linker=load('independent_linker');cs=[dict(s=1,d=3)]
        r=linker.annotate_joins(cs,np.array([[4.,1.],[.5,3.]]),[1,2],[3,4])[0]
        self.assertTrue(r['mutual']);self.assertEqual(r['row_margin'],3.);self.assertEqual(r['col_margin'],3.5)
        r=linker.annotate_joins(cs,np.array([[-1.]]),[1],[3])[0]
        self.assertEqual(r['row_margin'],-1.)

    def test_fork_veto_only_removes_heuristic_edge_with_independent_evidence(self):
        ns={0:node(0,0),1:node(1,1),2:node(1,2)};es=[edge(0,1,.9),edge(0,2,None)]
        cs=rules.fork_veto_candidates(ns,es);self.assertEqual(len(cs),1)
        self.assertEqual(rules.apply_fork_veto(ns,es,cs)[1],es)
        cs[0].update(weak_logit=-3.,keep_logit=4.,keep_top=True)
        nn,ee,st=rules.apply_fork_veto(ns,es,cs)
        self.assertEqual(nn,ns);self.assertEqual(ee,[es[0]]);self.assertEqual(st['fork_vetoes'],1)
        es[1]['edge_prob']=.5
        self.assertEqual(rules.apply_fork_veto(ns,es,cs)[1],es)

if __name__=='__main__': unittest.main()
