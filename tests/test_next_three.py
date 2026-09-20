import unittest
import numpy as np
import torch
from biohub_lab.resolution_probe import coarse_control,ResolutionProbe
from biohub_lab.dense_center import DenseCenter
from biohub_lab.uncertain_lineage import solve_events,reassign_uncertain

class NextThreeTests(unittest.TestCase):
    def test_zoom_coordinate_registration(self):
        from scipy.ndimage import zoom
        source=np.arange(17,dtype=float);resampled=zoom(source,.37,order=1)
        scale=(len(resampled)-1)/(len(source)-1)
        self.assertAlmostEqual(float(np.interp(8*scale,np.arange(len(resampled)),resampled)),8.)
        self.assertGreater(abs(float(np.interp(8*.37,np.arange(len(resampled)),resampled))-8.),.1)
    def test_coarse_control_preserves_sample_coordinates(self):
        x=torch.arange(16*64*64,dtype=torch.float32).reshape(1,1,16,64,64)
        y=coarse_control(x)
        torch.testing.assert_close(y[:,:,:,::4,::4],x[:,:,:,::4,::4],atol=.01,rtol=1e-6)
    def test_dense_logits_and_resolution_gradients(self):
        torch.set_num_threads(2)
        dense=DenseCenter();p=dense(torch.rand(1,1,16,16,16));self.assertEqual(p.shape,(1,16,16,16));p.mean().backward();self.assertIsNotNone(dense.first[0].weight.grad)
        model=ResolutionProbe();q=model(torch.rand(2,2,16,64,64));self.assertEqual(q.shape,(2,3));q.square().mean().backward();self.assertIsNotNone(model.net[0].weight.grad)
    def test_event_solver_relocates_split_without_double_parent(self):
        events=[(0,(0,1)),(0,(0,)),(1,(2,)),(1,(1,2))]
        chosen=solve_events(events,[4,0,4,0],2,3,1)
        self.assertEqual(set(chosen),{1,3})
    def test_infeasible_event_solver_does_not_return_invalid_graph(self):
        self.assertIsNone(solve_events([(0,(0,))],[0],1,2,0))
    def test_variable_division_count_can_remove_a_false_fork(self):
        events=[(0,(0,1)),(0,(0,)),(1,(2,)),(2,(1,))]
        chosen=solve_events(events,[4,0,0,0],3,3,None)
        self.assertEqual(set(chosen),{1,2,3})
    def test_uncertainty_preserves_nodes_child_capacity_and_forks(self):
        coords=np.array([[0,0,0,0],[0,0,20,0],[1,0,0,0],[1,0,2,0],[1,0,20,0]],float)
        ids=np.arange(5);edges=[(0,2),(0,3),(1,4)];features=np.eye(5)
        result,_=reassign_uncertain(ids,coords,edges,features)
        for e in result.values():
            self.assertEqual(sorted(b for a,b in e),[2,3,4]);self.assertEqual(len(e),3)

if __name__=='__main__':unittest.main()
