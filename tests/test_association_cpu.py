import unittest
import numpy as np
import torch
from biohub_lab.association_rank import AssociationRanker
from biohub_lab.association_cpu import calibrated_threshold, score_events
from biohub_lab.detection_dag import build_dag
from biohub_lab.event_solver import select_graph, validate_edges


class CPUAssociationTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_search_including_ties(self):
        s=np.array([3.,2.,2.,1.,0.]);y=np.array([0,1,0,1,0])
        for beta in (1.,.5):
            got=calibrated_threshold(s,y,beta);rows=[]
            for t in np.unique(s):
                p=s>=t;tp=int((p&(y==1)).sum());fp=int((p&(y==0)).sum());fn=int((~p&(y==1)).sum())
                value=(1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)
                rows.append((value,t))
            self.assertEqual((got['f_beta'],got['logit']),max(rows))

    def test_degenerate_calibration_rejected(self):
        with self.assertRaises(ValueError):calibrated_threshold([1,2],[1,1],1.)
        with self.assertRaises(ValueError):calibrated_threshold([1,float('nan')],[1,0],1.)

    def test_cpu_scoring_and_structured_graph_preserve_constraints(self):
        torch.manual_seed(5);torch.set_num_threads(2)
        coords=np.array([[0,8,20,20],[1,8,20,16],[1,8,20,24],[0,8,40,20],[1,8,40,20],[1,8,60,20]],np.int32)
        graph=build_dag(coords,np.ones(6),np.zeros(6,np.int8),(2,16,80,80))
        visual=np.zeros((6,64),np.float16)
        models={head:AssociationRanker(division=head=='division').eval() for head in ('edge','division')}
        scored,report=score_events(graph,visual,models,{'edge':{'logit':-10.},'division':{'logit':-10.}})
        self.assertEqual(report['scored_pairs'],int(graph['pair_counts'].sum()))
        self.assertTrue(np.isfinite(scored['edge_gains']).all())
        edges,solver=select_graph(graph,**scored)
        self.assertGreater(len(edges),0);validate_edges(coords,edges)
        scored.update(triples=np.empty((0,3),np.int64),triple_gains=np.empty(0,np.float32))
        edges,_=select_graph(graph,**scored)
        self.assertLessEqual(np.bincount(edges[:,0]).max(),1)


if __name__=='__main__':unittest.main()
