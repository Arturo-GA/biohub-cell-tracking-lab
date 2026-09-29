import collections
import importlib.util
import os
from pathlib import Path
import unittest
import ast
import json
import numpy as np
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/'kaggle/x138_xr'
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('context_rules')

def node(t,x):return dict(t=t,z=0.,y=0.,x=x/.40625)
def edge(s,d,p=.9,dist=2.):return dict(source_id=s,target_id=d,edge_prob=p,distance_um=dist)


class ContextTest(unittest.TestCase):
    def test_crossing_swap_and_degrees(self):
        ns={0:node(0,0),1:node(0,6),2:node(1,2),3:node(1,4),4:node(2,4),5:node(2,2),6:node(3,6),7:node(3,0)}
        es=[edge(s,d) for s,d in [(0,2),(1,3),(2,5),(3,4),(4,6),(5,7)]]
        n,out,st=m.temporal_swaps(ns,es)
        self.assertEqual(st['swap_pairs'],1);self.assertEqual(ns,n)
        self.assertIn((2,4),[(e['source_id'],e['target_id']) for e in out])
        for key in ['source_id','target_id']:
            self.assertEqual(collections.Counter(e[key] for e in es),collections.Counter(e[key] for e in out))
        self.assertEqual(m.temporal_swaps(ns,out)[2]['swap_pairs'],0)
        ns[8]=node(2,3);es.append(edge(2,8))
        self.assertEqual(m.temporal_swaps(ns,es)[2]['swap_pairs'],0)

    def test_end_rescue_requires_consistent_history(self):
        ns={i:node(i,i*2) for i in range(4)};es=[edge(i,i+1,.2 if i==2 else .9) for i in range(3)]
        self.assertEqual(m.context_cuts(ns,es)[2]['end_saved'],1)
        ns[3]=node(3,20)
        self.assertEqual(m.context_cuts(ns,es)[2]['end_saved'],0)
        self.assertEqual(len(m.context_cuts(ns,es)[1]),2)

    def test_long_rescue_and_noop_equivalence(self):
        ns={i:node(i,i*9) for i in range(5)};es=[edge(i,i+1,None,9) for i in range(4)]
        self.assertEqual(m.context_cuts(ns,es)[2]['long_saved'],1)
        self.assertEqual(len(m.context_cuts(ns,es,protect='off')[1]),0)
        xr=load('extra_rules');xr.xr_configure(cut_nan_dist_um=8,cut_end_prob=.5,cut_end_mode='last')
        ns={i:node(i,i*2) for i in range(6)}
        es=[edge(i,i+1,.2 if i==3 else .9,2) for i in range(5)];es[4]=edge(4,5,None,9)
        self.assertEqual(m.context_cuts(ns,es,protect='off')[1],xr.xr_pre_filter(ns,es)[1])

    def test_unknown_prefix_preserves_base_and_resets_override(self):
        with patch.dict(os.environ,{'BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU':'0.6','BIOHUB_XR_TAU_SPEC':'44b6:0.6,6bba:1.0'},clear=False):
            xr=load('extra_rules');xr.xr_set_safe_div_params('6bba_clip')
            self.assertEqual(xr.SAFE_DIV_SISTER_SYMMETRY_TAU,1.)
            xr.xr_set_safe_div_params('unknown_clip')
            self.assertEqual(xr.SAFE_DIV_SISTER_SYMMETRY_TAU,.6)

    def test_fork_pruning_boundary_is_optional_and_preserves_ilp(self):
        xr=load('extra_rules')
        xr.xr_configure(cut_nan_dist_um=0,cut_end_prob=0,fork_min_branch=8,
                        fork_nan_only='1',fork_preserve_boundary='0')
        ns={0:node(0,0),1:node(1,1),2:node(1,2),3:node(2,3)}
        es=[edge(0,1,None),edge(0,2),edge(2,3)]
        self.assertEqual(xr.xr_pre_filter(ns,es)[1],es[1:])
        xr.xr_configure(fork_preserve_boundary='1')
        self.assertEqual(xr.xr_pre_filter(ns,es)[1],es)
        # A different track extending the observed video gives enough future
        # frames: the original short repair branch must be removed again.
        ns[8]=node(8,0)
        self.assertEqual(xr.xr_pre_filter(ns,es)[1],es[1:])
        es[0]=edge(0,1,.2)
        self.assertEqual(xr.xr_pre_filter(ns,es)[1],es)
        stats={}
        self.assertEqual(xr.xr_pre_filter({},[],stats=stats),({},[]))
        self.assertEqual(stats['xr_fork_drops'],0)

    def test_empty_and_topology_smoothing(self):
        self.assertEqual(m.temporal_swaps({},[])[1],[])
        self.assertEqual(m.context_cuts({},[])[1],[])
        ns={0:node(0,0),1:node(1,1),2:node(1,2),3:node(2,3)}
        smooth={k:{**v,'x':v['x']+1} for k,v in ns.items()}
        out,n=m.topology_smoothing(ns,[edge(0,1),edge(0,2),edge(1,3)],smooth,weight=0)
        self.assertEqual(n,3)
        for k in [0,1,2]:self.assertEqual(out[k]['x'],ns[k]['x'])
        self.assertEqual(out[3]['x'],smooth[3]['x'])

    def test_smoothing_replay_matches_submission_reference(self):
        nb=json.loads((ROOT/'upstream/biohub-x138.ipynb').read_text(encoding='utf-8'))
        source=''.join(nb['cells'][5]['source'])
        fn=next(ast.get_source_segment(source,n) for n in ast.parse(source).body
                if isinstance(n,ast.FunctionDef) and n.name=='linefit_smooth_output_graph')
        scope=dict(np=np,OUTPUT_LINEFIT_SMOOTH=True,OUTPUT_LINEFIT_WINDOW=2,OUTPUT_LINEFIT_WEIGHT=.8)
        exec(fn,scope);replay=load('replay_linefit');rng=np.random.default_rng(43)
        for _ in range(10):
            ns={i:dict(t=i//3,z=float(rng.uniform(0,5)),y=2.5,x=float(rng.uniform(0,20))) for i in range(60)}
            es=[edge(i,i+3) for i in range(57)]+[edge(12,16),edge(24,28)]
            for w in (0.,.3):
                ref=scope['linefit_smooth_output_graph']({k:dict(v) for k,v in ns.items()},es,collections.Counter())
                ref,_=m.topology_smoothing(ns,es,ref,w)
                plain=replay.full_linefit_round(ns,es,raw_output=True,tie_weights=(w,))
                got={k:dict(t=v[0],z=v[1],y=v[2],x=v[3]) for k,v in plain.items()}
                got,_=m.topology_smoothing(ns,es,got,w)
                self.assertEqual({k:replay.rounded_node(v) for k,v in ref.items()},
                                 {k:replay.rounded_node(v) for k,v in got.items()})


if __name__=='__main__':unittest.main()
