import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from biohub_lab.identity_parent import parent_table,supervision,decode,ParentModel

class IdentityParentTests(unittest.TestCase):
    def test_unique_supervision_masks_duplicate_unknown(self):
        coords=np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0]],np.float32)
        truth=coords[[0,2]];parents=parent_table(coords,k=3)
        d=supervision(coords,parents,truth,np.array([[0,1]]))
        self.assertEqual(d['targets'].tolist(),[2])
        self.assertEqual(int(d['masks'][0,:-1].sum()),1)
        self.assertTrue(d['masks'][0,d['labels'][0]])
        self.assertNotEqual(d['labels'][0],3)

    def test_null_when_true_parent_not_in_candidate_set(self):
        coords=np.array([[0,0,0,0],[0,0,0,30],[1,0,0,1]],np.float32)
        parents=parent_table(coords,k=1)
        d=supervision(coords,parents,coords,np.array([[1,2]]))
        self.assertEqual(d['labels'].tolist(),[1]);self.assertEqual(d['null_examples'],1)

    def test_global_assignment_capacity_and_null(self):
        coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,10]],np.float32)
        parents=np.array([[-1,-1],[-1,-1],[0,1],[0,1]])
        logits=np.array([[-1e4,-1e4,0],[-1e4,-1e4,0],[10,9,0],[8,-1,0]])
        self.assertEqual(set(map(tuple,decode(coords,parents,logits))),{(1,2),(0,3)})
        self.assertEqual(len(decode(coords,parents,np.c_[np.full((4,2),-1),np.zeros(4)])),0)

    def test_cpu_training_and_reload(self):
        from identity_parent_runner import train
        torch.set_num_threads(2)
        coords=np.array([[0,0,0,0],[0,0,0,20],[1,0,0,1],[1,0,0,19]],np.float32)
        p=parent_table(coords,k=3);d=supervision(coords,p,coords,np.array([[0,2],[1,3]]))
        v=dict(coords=coords,nodes=torch.randn(4,68),positions=torch.tensor(coords[:,1:]/20),parents=torch.tensor(p),labels=d)
        with tempfile.TemporaryDirectory() as folder:
            model=train({'synthetic':v},Path(folder),steps=3)
            with torch.no_grad():expected=model(v['nodes'],v['positions'],v['parents'],torch.tensor([2,3]))
            other=ParentModel();other.load_state_dict(torch.load(Path(folder)/'last.pt',weights_only=False)['state_dict'])
            with torch.no_grad():actual=other(v['nodes'],v['positions'],v['parents'],torch.tensor([2,3]))
            self.assertTrue(torch.isfinite(actual).all());torch.testing.assert_close(actual,expected)

if __name__=='__main__':unittest.main()
