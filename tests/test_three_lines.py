import numpy as np
import torch
import unittest
from biohub_lab.temporal_detector import CenterField,vector_targets,votes

def test_unknown_voxels_have_no_supervision_and_nearest_overlap():
    y,m=vector_targets(np.array([[8.,8.,8.],[11.,8.,8.]]),(20,20,20),radius=3)
    assert not m[0,0,0] and not m[19,19,19]
    np.testing.assert_allclose(y[:,9,8,8],[-1,0,0])
    np.testing.assert_allclose(y[:,10,8,8],[1,0,0])
    assert m.sum()<20**3//5

def test_vector_voting_recovers_center():
    xyz=np.indices((16,16,16),dtype=np.float32);center=np.array([8.,7.,9.])[:,None,None,None]
    field=np.clip(center-xyz,-4,4);image=np.exp(-np.square(xyz-center).sum(0)/4)
    points,scores=votes(field,image)
    np.testing.assert_allclose(points[0],[8,7,9]);assert scores[0]>0

def test_center_network_dense_shape_and_gradient():
    torch.set_num_threads(2);model=CenterField(3);x=torch.randn(1,3,16,16,16);y=model(x)
    assert y.shape==(1,3,16,16,16) and float(y.detach().abs().max())<=4
    y.square().mean().backward();assert model.first[0].weight.grad.abs().sum()>0

def test_expanded_events_retain_short_context_and_known_negatives():
    from biohub_lab.expanded_events import event_candidates
    coords=np.array([[0,8,8,8],[1,7,8,8],[1,9,8,8],[0,8,12,8],[1,8,11,8]],float)
    edges=np.array([[0,1],[0,2],[3,4]])
    t,y=event_candidates(coords,edges)
    assert any(tuple(a)==(0,1,2) and b==1 for a,b in zip(t,y))
    assert y.sum()==1 and (y==0).sum()>0

def test_event_query_daughter_symmetry_and_reflection():
    from biohub_lab.expanded_events import query_masks,geometry,JointEventModel
    delta=np.array([[[2.,0.,0.],[-2.,1.,0.]]],np.float32)
    a=query_masks(delta,'cpu');b=query_masks(delta[:,::-1].copy(),'cpu');torch.testing.assert_close(a,b)
    np.testing.assert_allclose(geometry(delta),geometry(delta[:,::-1]))
    flipped=delta.copy();flipped[:,:,0]=-flipped[:,:,0]-1
    c=query_masks(flipped,'cpu',np.array([[-1.,0.,0.]],np.float32));torch.testing.assert_close(c,a.flip(2),atol=1e-6,rtol=1e-6)
    m=JointEventModel();x=torch.cat([torch.randn(1,5,32,32,32),a],1);y=m(x,torch.as_tensor(geometry(delta)));assert y.shape==(1,)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f,description=name)
        for name,f in globals().items() if name.startswith('test_'))

if __name__=='__main__':unittest.main()
