import unittest
import numpy as np
from biohub_lab.joint_division import select,proposals

class JointDivisionTests(unittest.TestCase):
    def test_parent_reassignment_and_global_conflict(self):
        coords=np.array([[0,1,1,1],[0,1,1,2],[1,1,1,1],[1,1,1,2]])
        edges=np.array([[0,2],[1,3]])
        event=dict(kind='parent',gain=5,remove=[(1,3)],add=[(0,3)],dense=[],resources=[0,1,2,3])
        loser=dict(event,gain=2)
        c,e,r,chosen=select(coords,edges,np.empty((0,4)),[event,loser],False)
        self.assertEqual(set(map(tuple,e)),{(0,2),(0,3)});self.assertEqual(r['selected'],1);self.assertEqual(chosen[0]['gain'],5)
    def test_missing_node_chain_and_ablation(self):
        coords=np.array([[0,1,1,1],[1,1,1,1],[2,1,1,2]])
        edges=np.array([[0,1]]);dense=np.array([[1,1,1,2]])
        event=dict(kind='donor',gain=5,remove=[],add=[(0,3),(3,2)],dense=[0],resources=[0,1,2,3])
        c,e,r,_=select(coords,edges,dense,[event],True)
        self.assertEqual(len(c),4);self.assertEqual(set(map(tuple,e)),{(0,1),(0,3),(3,2)})
        self.assertEqual(select(coords,edges,dense,[event],False)[2]['selected'],0)
    def test_no_history_does_not_propose(self):
        c=np.array([[0,1,1,1],[1,1,1,1]]);events,r=proposals(c,np.array([[0,1]]),np.empty((0,4)))
        self.assertEqual(events,[])
    def test_real_proposal_generation_for_split_and_missing_daughter(self):
        c=np.array([[0,10,20,20],[1,10,20,20],[2,10,20,12],[3,10,20,10],[4,10,20,8],[2,10,20,28],[3,10,20,30],[4,10,20,32],[0,10,20,50],[1,10,20,50]],float)
        edges=np.array([[0,1],[1,2],[2,3],[3,4],[9,5],[5,6],[6,7],[8,9]])
        c=np.vstack([c,[[t,10,20,60] for t in range(6)]])
        edges=np.vstack([edges,[[i,i+1] for i in range(10,15)]])
        events,_=proposals(c,edges,np.empty((0,4)))
        self.assertTrue(any(e['mother']==1 and e['gain']>1 and (9,5) in e['remove'] for e in events))
        keep=[0,1,2,3,4,6,7,*range(10,16)];mapping={n:i for i,n in enumerate(keep)}
        reduced=np.array([[mapping[a],mapping[b]] for a,b in edges if a in mapping and b in mapping])
        events,_=proposals(c[keep],reduced,c[[5]])
        self.assertTrue(any(e['mother']==1 and e['kind']=='donor' and e['gain']>1 for e in events))
if __name__=='__main__':unittest.main()
