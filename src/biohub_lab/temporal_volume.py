"""E033 trainable volumetric temporal representation and annotation-safe candidates."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALE=np.array([1.625,.40625,.40625])

def records(coords,truth,edges,mapping):
    from .identity_parent import parent_table
    gt={int(r[0]):r for r in truth};reverse={p:g for g,p in mapping.items()}
    parents=parent_table(coords,12,20.);indices=[];labels=[];valids=[];masks=[];distances=[]
    for a,b in edges:
        if a not in mapping or b not in mapping:continue
        p=parents[mapping[b]];valid=p>=0;correct=np.flatnonzero(p==mapping[a])
        if not len(correct) or valid.sum()<2:continue
        safe=np.maximum(p,0)
        far=np.linalg.norm((coords[safe,1:]-gt[a][2:])*SCALE,axis=1)>7
        known_other=np.array([j in reverse and reverse[j]!=a for j in p])
        mask=valid&(far|known_other|(p==mapping[a]))
        indices.append(np.r_[mapping[b],safe]);labels.append(int(correct[0]));valids.append(valid);masks.append(mask)
        distances.append(np.linalg.norm((coords[safe,1:]-coords[mapping[b],1:])*SCALE,axis=1))
    return dict(indices=np.asarray(indices,np.int64).reshape(-1,13),labels=np.asarray(labels,np.int64),valid=np.asarray(valids,bool).reshape(-1,12),mask=np.asarray(masks,bool).reshape(-1,12),distance=np.asarray(distances,np.float32).reshape(-1,12))

def crop_temporal(padded,coord,size=16):
    """Padded TZYX image at isotropic1.625um; three time points at same position."""
    t,z,y,x=np.rint(coord/np.array([1,1,4,4])).astype(int)
    times=np.clip(np.array([t-1,t,t+1]),0,len(padded)-1)
    out=padded[times,z:z+size,y:y+size,x:x+size]
    if out.shape!=(3,size,size,size):raise ValueError('Crop out of bounds')
    return out

def gather_temporal(padded,centers):
    """Vectorized extraction identical to crop_temporal for already scaled centers."""
    axis=torch.arange(16,device=padded.device);delta=torch.tensor([-1,0,1],device=padded.device)
    t=(centers[:,0,None]+delta).clamp(0,len(padded)-1)
    z=centers[:,1,None]+axis;y=centers[:,2,None]+axis;x=centers[:,3,None]+axis
    return padded[t[:,:,None,None,None],z[:,None,:,None,None],y[:,None,None,:,None],x[:,None,None,None,:]]

class TemporalVolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(3,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv3d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv3d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(64,64))
    def forward(self,x):return F.normalize(self.net(x),dim=-1)

def logits(embeddings,distance,temperature=.2):
    cosine=(embeddings[:,:1]*embeddings[:,1:]).sum(-1)
    return cosine/temperature-distance/5.,cosine/temperature
