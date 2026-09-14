from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile
import unittest
import numpy as np
from biohub_lab.detection_dag import (DAGConfig,build_dag,merge_sources,pair_allowed,two_paths,
    select_development,save_dag)
from biohub_lab.dag_coverage import evaluate_graph,readiness,CoverageGate


def simple_graph():
    # Mother appears at the first frame. Neither daughter has an old track.
    coords=np.array([[0,8,40,40]]+[[t,8,40,x] for t in range(1,5) for x in (32,48)],np.int32)
    config=replace(DAGConfig(),sample_queries=8)
    graph=build_dag(coords,np.ones(len(coords)),np.ones(len(coords)),(5,24,96,96),config)
    edges=np.array([[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]])
    return graph,coords,edges,config


class DetectionDAGTests(unittest.TestCase):
    def test_first_frame_mother_and_two_unlinked_daughters_generate_paths(self):
        graph,coords,edges,config=simple_graph()
        self.assertTrue(pair_allowed(graph,0,1,2))
        paths=two_paths(graph,1,2,4)
        self.assertIsNotNone(paths);self.assertFalse(set(paths[0])&set(paths[1]))
        np.testing.assert_array_equal(graph['coords'][paths[0],0],[1,2,3,4])
        counts,records=evaluate_graph(graph,coords,edges,config,match_um=.1)
        self.assertEqual(counts,dict(events=1,centers_covered=1,pairs_covered=1,paths_covered=1,full_context_events=1,annotated_paths_covered=1))
        with tempfile.TemporaryDirectory() as folder:
            result=save_dag(graph,folder,'synthetic',config)
            self.assertFalse(result['annotations_read']);self.assertTrue((Path(folder)/'trajectory_samples.npz').exists())

    def test_residual_flow_reroutes_a_greedy_collision(self):
        graph=dict(coords=np.array([[1,0,0,0],[1,0,0,1],[2,0,0,0],[2,0,0,1],[3,0,0,0],[3,0,0,1]]),
            continuation=np.array([[2,3],[2,-1],[4,-1],[5,-1],[-1,-1],[-1,-1]]),shape=np.array([4,1,1,2]))
        self.assertEqual(two_paths(graph,0,1,3),[[0,3,5],[1,2,4]])

    def test_flow_matches_exhaustive_small_graphs(self):
        rng=np.random.default_rng(1729)
        coords=np.array([[t,0,0,n] for t in range(4) for n in range(3)])
        def paths(table,node):
            if coords[node,0]==3:return [[node]]
            return [[node]+p for child in table[node] if child>=0 for p in paths(table,int(child))]
        for _ in range(25):
            table=np.full((12,3),-1)
            for node in range(9):
                targets=np.arange(3*(coords[node,0]+1),3*(coords[node,0]+2))
                keep=targets[rng.random(3)<.55];table[node,:len(keep)]=keep
            possible=any(not set(a)&set(b) for a,b in product(paths(table,0),paths(table,1)))
            result=two_paths(dict(coords=coords,continuation=table,shape=np.array([4,1,1,3])),0,1,3)
            self.assertEqual(result is not None,possible)

    def test_missing_detection_pair_cap_and_continuation_fail_at_distinct_stages(self):
        graph,truth,edges,config=simple_graph()
        # No second daughter detection anywhere in the volume.
        single=build_dag(truth[[0,1,3,5,7]],np.ones(5),np.ones(5),(5,24,96,96),config)
        counts,rows=evaluate_graph(single,truth,edges,config,match_um=.1)
        self.assertEqual(rows[0]['reason'],'missing_detection')
        narrow=replace(config,max_daughters=2,mother_radius_um=1.)
        none=build_dag(truth,np.ones(9),np.ones(9),(5,24,96,96),narrow)
        counts,rows=evaluate_graph(none,truth,edges,narrow,match_um=.1)
        self.assertEqual(rows[0]['reason'],'initial_pair_pruned')
        broken={key:value.copy() for key,value in graph.items()};broken['continuation'][[1,2]]=-1
        counts,rows=evaluate_graph(broken,truth,edges,config,match_um=.1)
        self.assertEqual(rows[0]['reason'],'no_disjoint_continuations')

    def test_two_geometric_paths_are_not_automatically_correct_annotated_paths(self):
        graph,truth,edges,config=simple_graph()
        graph['continuation'][1]=np.array([4]+[-1]*7) # force first daughter to switch identity
        graph['continuation'][2]=np.array([3]+[-1]*7)
        counts,_=evaluate_graph(graph,truth,edges,config,match_um=.1)
        self.assertEqual(counts['paths_covered'],1)
        self.assertEqual(counts['annotated_paths_covered'],0)

    def test_selection_is_filename_only_disjoint_and_order_invariant(self):
        names=[f'{group}_{i:04}' for group in ('a','b') for i in range(40)]
        excluded=['a_0000','b_0000'];a=select_development(names,excluded)
        self.assertEqual(a,select_development(names[::-1],excluded));self.assertEqual(len(a),48)
        self.assertFalse(set(a)&set(excluded))

    def test_merge_keeps_native_centers_and_rejects_label_like_fractional_time(self):
        source=np.array([[0,8,40,32],[0,8,40,32.4],[0,8,40,48]],np.float32)
        coords,scores,origin=merge_sources([(source,np.array([.8,.5,.7]))],(5,24,96,96))
        self.assertEqual(len(coords),2);self.assertTrue(np.all(coords==np.rint(coords)))
        source[0,0]=.5
        with self.assertRaises(ValueError):merge_sources([(source,np.ones(3))],(5,24,96,96))

    def test_small_or_low_coverage_audit_cannot_pass_readiness(self):
        counts=dict(events=1,centers_covered=1,pairs_covered=1,paths_covered=1,full_context_events=1,annotated_paths_covered=1)
        self.assertFalse(readiness({'a_1':counts,'b_1':counts})['coverage_gate_passed'])
        large={k:20*v for k,v in counts.items()}
        result=readiness({'a_1':large,'b_1':large});self.assertTrue(result['coverage_gate_passed'])
        large=dict(large,annotated_paths_covered=0)
        self.assertFalse(readiness({'a_1':large,'b_1':large})['coverage_gate_passed'])
        self.assertFalse(result['accuracy_demonstrated']);self.assertFalse(result['training_started'])


if __name__=='__main__':unittest.main()
