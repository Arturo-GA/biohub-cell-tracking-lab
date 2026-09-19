"""Unique-identity supervision and competing parents on fixed primary detections."""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .association_rank import AssociationRanker,SCALE
from .detection_identity import mappings,maximum_vote_edges

def parent_table(coords,k=12,radius=20.):
    table=np.full((len(coords),k),-1,np.int64)
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t-1);b=np.flatnonzero(coords[:,0]==t)
        if not len(a):continue
        d,ix=cKDTree(coords[a,1:]*SCALE).query(coords[b,1:]*SCALE,k=k,distance_upper_bound=radius)
        d,ix=d.reshape(len(b),k),ix.reshape(len(b),k)
        rows,cols=np.nonzero(np.isfinite(d));table[b[rows],cols]=a[ix[rows,cols]]
    return table

def supervision(coords,parents,truth,edges):
    _,mapping=mappings(coords,truth)
    gt_parent=np.full(len(truth),-1,np.int64)
    for a,b in edges:
        if gt_parent[b]>=0:raise ValueError('Multiple annotated parents')
        gt_parent[b]=a
    targets=[];labels=[];masks=[]
    for target in np.flatnonzero(mapping>=0):
        true=gt_parent[mapping[target]]
        if true<0:continue
        candidates=parents[target];valid=candidates>=0
        mapped=np.full(len(candidates),-1,np.int64);mapped[valid]=mapping[candidates[valid]]
        mask=mapped>=0 # Unannotated candidates are unknown, never negative examples.
        correct=np.flatnonzero(mask & (mapped==true))
        assert len(correct)<=1
        # Null means no represented true parent in this candidate set, not cell background.
        label=int(correct[0]) if len(correct) else len(candidates)
        if not mask.any():continue
        targets.append(target);labels.append(label);masks.append(np.r_[mask,True])
    return dict(targets=np.asarray(targets,np.int64),labels=np.asarray(labels,np.int64),
        masks=np.asarray(masks,bool).reshape(-1,parents.shape[1]+1),
        matched_nodes=int((mapping>=0).sum()),null_examples=int(np.sum(np.asarray(labels)==parents.shape[1])))

class ParentModel(nn.Module):
    def __init__(self):
        super().__init__();self.pair=AssociationRanker();self.null=nn.Parameter(torch.zeros(()))
    def forward(self,nodes,positions,parents,targets):
        p=parents[targets];safe=p.clamp_min(0)
        child=targets[:,None].expand_as(p)
        pairs=torch.stack((safe,child),-1)
        scores=self.pair(nodes[pairs.reshape(-1,2)],positions[pairs.reshape(-1,2)]).reshape(p.shape)
        scores=scores.masked_fill(p<0,-1e4)
        return torch.cat((scores,self.null.expand(len(targets),1)),1)

def decode(coords,parents,logits):
    rows,cols=np.nonzero(parents>=0)
    gain=logits[rows,cols]-logits[rows,-1]
    keep=gain>0
    edges=np.column_stack((parents[rows[keep],cols[keep]],rows[keep]))
    # maximum_vote_edges adds a fixed <=.01 geometric tie break.
    return maximum_vote_edges(coords,edges,gain[keep])

def geometric_logits(coords,parents):
    safe=np.maximum(parents,0)
    dist=np.linalg.norm((coords[safe,1:]-coords[:,None,1:])*SCALE,axis=2)
    return np.c_[np.where(parents>=0,3.-dist/5.,-1e4),np.zeros(len(coords))].astype(np.float32)
