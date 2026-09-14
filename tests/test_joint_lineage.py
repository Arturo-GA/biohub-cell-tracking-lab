from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile
import unittest
import numpy as np
from biohub_lab.joint_lineage import (JointConfig,ImageEvidence,enumerate_windows,
    reconstruct,solve_event_packing,graph_index,apply_events)
from biohub_lab.detector_proposals import SCALE


def scene(independent=False):
    shape=(10,24,96,128)
    primary=np.array([[t,12,48,64 if t<6 else 56] for t in range(9)],np.int64)
    secondary=np.array([[t,12,48,72] for t in range(5,9)],np.int64)
    coords=np.concatenate([primary,secondary])
    edges=np.array([(i,i+1) for i in range(8)]+[(i,i+1) for i in range(9,12)],np.int64)
    proposals=np.array([[t,12,48,x] for t in range(3,6) for x in (56,72)],np.float32)
    axes=[np.arange(n)*s for n,s in zip((shape[1],shape[2]//2,shape[3]//2),ImageEvidence.spacing)]
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    center=np.array([12,48,64])*SCALE
    def spot(point):return np.exp(-.5*np.sum(((grid-point)/1.5)**2,axis=-1))
    one=spot(center)
    two=.5*(spot(np.array([12,48,56])*SCALE)+spot(np.array([12,48,72])*SCALE))
    volume=np.stack([two if independent or t>=3 else one for t in range(shape[0])]).astype(np.float32)
    return coords,edges,proposals,shape,ImageEvidence(volume)


class JointTests(unittest.TestCase):
    def test_joint_reconstruction_replaces_merged_centers_and_recovers_division(self):
        base,edges,proposal,shape,evidence=scene()
        with tempfile.TemporaryDirectory() as folder:
            coords,linked,receipt=reconstruct(base,edges,[(proposal,np.ones(len(proposal)))],shape,evidence,folder)
            self.assertEqual(receipt['selected_events'],1)
            self.assertGreater(receipt['changes']['added_nodes'],0)
            self.assertGreater(receipt['changes']['removed_base_nodes'],0)
            children,parents=graph_index(coords,linked)
            mother=int(np.flatnonzero((coords[:,0]==2)&(coords[:,3]==64))[0])
            self.assertEqual(len(children[mother]),2)
            self.assertEqual(set(coords[children[mother],3]),{56,72})
            self.assertFalse(np.any((coords[:,0]>=3)&(coords[:,0]<=5)&(coords[:,3]==64)))
            self.assertTrue((Path(folder)/'window_audit.jsonl').exists())
            self.assertTrue((Path(folder)/'event_audit.jsonl').exists())

    def test_two_preexisting_nuclei_do_not_become_a_division(self):
        base,edges,proposal,shape,evidence=scene(independent=True)
        with tempfile.TemporaryDirectory() as folder:
            coords,linked,receipt=reconstruct(base,edges,[(proposal,np.ones(len(proposal)))],shape,evidence,folder)
        self.assertEqual(receipt['selected_events'],0)
        np.testing.assert_array_equal(coords,base)
        self.assertEqual(set(map(tuple,linked)),set(map(tuple,edges)))
        self.assertGreater(receipt['pair_decisions'].get('already_two_image_peaks',0),0)

    def test_no_proposals_retains_original_graph(self):
        base,edges,_,shape,evidence=scene()
        with tempfile.TemporaryDirectory() as folder:
            coords,linked,receipt=reconstruct(base,edges,[],shape,evidence,folder)
        np.testing.assert_array_equal(coords,base)
        self.assertEqual(set(map(tuple,linked)),set(map(tuple,edges)))

    def test_packing_beats_greedy_and_matches_exhaustive(self):
        pool=np.array([[t,0,0,20*i] for i,t in enumerate([4,4,4,4,4,4])])
        events=[dict(resources=r,paths=[[2*i],[2*i+1]],evidence={'gain':g})
                for i,(r,g) in enumerate([([10,11],7.),([10],4.),([11],4.)])]
        selected,receipt=solve_event_packing(events,pool)
        self.assertEqual(set(selected),{1,2})
        feasible=[]
        for mask in product((0,1),repeat=3):
            resources=[r for i,e in enumerate(events) if mask[i] for r in e['resources']]
            if len(resources)==len(set(resources)):
                feasible.append(sum(e['evidence']['gain'] for i,e in enumerate(events) if mask[i]))
        self.assertEqual(sum(events[i]['evidence']['gain'] for i in selected),max(feasible))

    def test_spatially_duplicate_proposals_conflict_across_events(self):
        pool=np.array([[3,10,20,20],[4,10,20,20],[3,10,20,21],[4,10,20,22],
                       [3,20,20,20],[4,20,20,20],[3,30,20,20],[4,30,20,20]])
        events=[dict(resources=[100],paths=[[0,1],[4,5]],evidence={'gain':2.}),
                dict(resources=[200],paths=[[2,3],[6,7]],evidence={'gain':3.})]
        selected,_=solve_event_packing(events,pool)
        self.assertEqual(selected,[1])

    def test_candidates_require_persistence_and_consecutive_graph(self):
        base,edges,_,_,_=scene()
        self.assertGreater(len(enumerate_windows(base,edges)),0)
        # Removing the orphan's later continuation prevents it becoming an anchor.
        short=edges[edges[:,0]!=11]
        self.assertEqual(enumerate_windows(base,short),[])
        invalid=np.array([[0,2]])
        with self.assertRaises(ValueError):graph_index(base,invalid)


if __name__=='__main__':unittest.main()
