"""Joint mother/daughter image queries from complete available sparse lineage labels."""
import itertools
import numpy as np
import torch
from torch import nn

def event_candidates(coords,edges,training=False,seed=400920):
    children={};parent={}
    for a,b in edges:children.setdefault(int(a),[]).append(int(b));parent[int(b)]=int(a)
    triples=[];labels=[]
    for a,kids in sorted(children.items()):
        assert len(kids)<=2
        known=np.array([b for b in parent if coords[b,0]==coords[a,0]+1],int)
        dist=np.linalg.norm(coords[known,1:]-coords[a,1:],axis=1)*1.625
        near=known[np.argsort(dist,kind='stable')[:6]];near=[int(b) for b in near if np.linalg.norm(coords[b,1:]-coords[a,1:])*1.625<=20]
        # Include every true division, even if candidate distance would miss it.
        pairs=set(itertools.combinations(sorted(set(near+kids)),2))
        for b,c in sorted(pairs):
            positive=len(kids)==2 and set(kids)=={b,c}
            negative=parent[b]!=a or parent[c]!=a
            if positive or negative:triples.append([a,b,c]);labels.append(int(positive))
    triples=np.array(triples,np.int64).reshape(-1,3);labels=np.array(labels,np.int64)
    if training:
        positive=np.flatnonzero(labels==1);negative=np.flatnonzero(labels==0);rng=np.random.default_rng(seed)
        if len(negative)>128:negative=np.sort(rng.choice(negative,128,replace=False))
        keep=np.sort(np.r_[positive,negative]);triples,labels=triples[keep],labels[keep]
    return triples,labels

def geometry(delta):
    a,b=delta[:,0],delta[:,1];da=np.linalg.norm(a,axis=1);db=np.linalg.norm(b,axis=1)
    return np.c_[np.minimum(da,db),np.maximum(da,db),np.linalg.norm(a-b,axis=1),np.linalg.norm((a+b)/2,axis=1),np.abs(a+b)/2,np.abs(a-b),(a*b).sum(1)]/16

def query_masks(delta,device,mother_offset=None):
    grid=torch.stack(torch.meshgrid(*(torch.arange(32,device=device) for _ in range(3)),indexing='ij')).float()
    center=torch.tensor([16.,16.,16.],device=device)
    offsets=torch.zeros((len(delta),3),device=device) if mother_offset is None else torch.as_tensor(mother_offset,device=device)
    mother=torch.exp(-((grid[None]-(center+offsets)[:,:,None,None,None])**2).sum(1)/2)
    points=torch.as_tensor(delta,device=device)+center
    daughter=torch.exp(-((grid[None,None]-points[:,:,:,None,None,None])**2).sum(2)/2).sum(1).clamp_max(1)
    return torch.stack([mother,daughter],1)

class JointEventModel(nn.Module):
    def __init__(self):
        super().__init__();layers=[];c=7
        for width in [16,32,64,96]:layers.extend([nn.Conv3d(c,width,3,stride=2,padding=1),nn.GroupNorm(4,width),nn.SiLU()]);c=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool3d(1),nn.Flatten());self.head=nn.Sequential(nn.Linear(107,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x,g):return self.head(torch.cat([self.encoder(x),g],1)).squeeze(1)
