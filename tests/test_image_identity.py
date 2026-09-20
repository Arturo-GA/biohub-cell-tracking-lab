"""Behavioral tests for image geometry, real-token extraction and joint decoding."""
import numpy as np
import torch
from torch import nn
from biohub_lab.image_identity import physical_views,extract_tokens,ImageIdentityModel,decode_events


def test_physical_sections_respect_anisotropy():
    z,y,x=np.indices((20,40,40));volume=z*1.625+y*.40625+x*.40625
    views=physical_views(volume,np.array([10,20,20]),size=8)
    # Linear isotropic physical field has the same sampled slope in all planes.
    for view in views:
        np.testing.assert_allclose(np.diff(view,axis=0),.40625,atol=1e-5)
        np.testing.assert_allclose(np.diff(view,axis=1),.40625,atol=1e-5)


def test_real_tokens_ignore_random_styles():
    class MockNet(nn.Module):
        ps=8
        def __init__(self):super().__init__();self.out=nn.Linear(4,3)
        def forward(self,x):
            tokens=torch.arange(64*4,dtype=x.dtype).reshape(1,64,4).expand(len(x),-1,-1)
            return self.out(tokens),torch.randn(len(x),256)
    net=MockNet().eval();x=torch.ones(2,1,64,64)
    a=extract_tokens(net,x);b=extract_tokens(net,x)
    assert a.shape==(2,8) and torch.equal(a,b)
    assert not net.out._forward_pre_hooks


def test_division_and_identity_symmetry():
    torch.manual_seed(1);model=ImageIdentityModel(features=12,hidden=8).eval()
    h=model.encode(torch.randn(3,12));coords=torch.tensor([[0.,0,0,0],[1,1,2,3],[1,2,3,4]])
    a=model.division_logits(h,coords,torch.tensor([[0,1,2]]));b=model.division_logits(h,coords,torch.tensor([[0,2,1]]))
    torch.testing.assert_close(a,b)
    a=model.identity_logits(h,coords,torch.tensor([[1,2]]));b=model.identity_logits(h,coords,torch.tensor([[2,1]]))
    torch.testing.assert_close(a,b)


def test_decoder_joint_division_and_duplicate_competition():
    coords=np.array([[0,0,0,0],[0,0,0,1],[1,0,1,0],[1,0,-1,0]],float)
    edges=np.array([[0,2],[0,3],[1,2],[1,3]])
    # Independent edge selection would take two separate mothers. One joint
    # division wins with its score; same-frame source duplicates cannot coexist.
    selected,report=decode_events(coords,edges,np.array([2.,2.,3.,3.]),np.array([[0,2,3]]),np.array([7.]),[(0,1)])
    assert selected.tolist()==[[0,2],[0,3]] and report['divisions']==1
    selected,_=decode_events(coords,edges,np.full(4,-1.),np.empty((0,3),int),np.empty(0),[(0,1)])
    assert selected.shape==(0,2)


def test_appearance_can_change_equal_geometry_scores():
    torch.manual_seed(3);model=ImageIdentityModel(features=12,hidden=8)
    coords=torch.zeros(3,4);coords[1:,0]=1
    features=torch.randn(3,12,requires_grad=True)
    score=model.edge_logits(model.encode(features),coords,torch.tensor([[0,1],[0,2]]))
    assert not torch.allclose(score[0],score[1])
    score.sum().backward();assert features.grad.abs().sum()>0


def test_learns_identity_when_distance_cannot_disambiguate():
    # Deliberately equal positions: only paired appearance identifies parents.
    torch.set_num_threads(2);torch.manual_seed(19)
    model=ImageIdentityModel(features=12,hidden=16)
    base=torch.randn(6,12);features=torch.cat((base,base+.01*torch.randn_like(base)))
    coords=torch.zeros(12,4);coords[6:,0]=1
    edges=torch.tensor([[a,b] for b in range(6,12) for a in range(6)])
    labels=torch.arange(6);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(100):
        logits=model.edge_logits(model.encode(features),coords,edges).reshape(6,6)
        loss=torch.nn.functional.cross_entropy(logits,labels)
        optimizer.zero_grad();loss.backward();optimizer.step()
    predicted=model.edge_logits(model.encode(features),coords,edges).reshape(6,6).argmax(1)
    assert torch.equal(predicted,labels)


def test_decoder_refuses_nonconsecutive_or_invalid_duplicate_events():
    coords=np.array([[0,0,0,0],[2,0,0,0]],float)
    try:decode_events(coords,np.array([[0,1]]),[1.],np.empty((0,3),int),[])
    except ValueError:pass
    else:raise AssertionError('Skipped frame accepted')
