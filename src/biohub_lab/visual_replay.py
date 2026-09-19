"""Replay a public pretrained association head on cached image embeddings.

This is an approximation: E013 sampled subvoxels from first-seen windows;
the baseline sampled integer locations from pair-specific feature maps.
"""
import importlib.util
import numpy as np
import torch
from scipy.spatial import cKDTree
from .association_rank import SCALE

def load_head(source,checkpoint):
    spec=importlib.util.spec_from_file_location('frozen_public_transformer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    weights={k.removeprefix('transformer.'):v for k,v in state.items() if k.startswith('transformer.')}
    assert weights and weights['proj.weight'].shape[1]==64
    model=module.SimpleNodeTransformer(feat_dim=64)
    model.load_state_dict(weights,strict=True);model.eval();return model

def position(coords,shape,time):
    points=coords.astype(float).copy();points[:,0]=time
    points[:,1:]/=np.array([1,4,4])
    extent=np.array([2,shape[1],(shape[2]+3)//4,(shape[3]+3)//4])
    angles=points[:,:,None]/extent[None,:,None]*2.**np.arange(4)*np.pi
    return np.concatenate((np.sin(angles),np.cos(angles)),axis=2).reshape(len(points),32).astype('float32')

def replay(nodes,graph,features,model):
    ids=sorted(nodes);coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids])
    mapping={}; distances=[]
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero((graph['coords'][:,0]==t)&(graph['origin']==0))
        if not len(b):continue
        d,j=cKDTree(graph['coords'][b,1:]*SCALE).query(coords[a,1:]*SCALE)
        _,back=cKDTree(coords[a,1:]*SCALE).query(graph['coords'][b,1:]*SCALE)
        good=(d<=1.)&(back[j]==np.arange(len(a)))
        for x,y,dist in zip(a[good],b[j[good]],d[good]):mapping[int(x)]=int(y);distances.append(float(dist))
    output={};frames=0
    with torch.inference_mode():
        for t in np.unique(coords[:,0]):
            a=np.array([i for i in np.flatnonzero(coords[:,0]==t) if i in mapping],int)
            b=np.array([i for i in np.flatnonzero(coords[:,0]==t+1) if i in mapping],int)
            if not len(a) or not len(b):continue
            fa=torch.tensor(np.c_[features[[mapping[i] for i in a],32:].astype('float32'),position(coords[a],graph['shape'],0)])
            fb=torch.tensor(np.c_[features[[mapping[i] for i in b],32:].astype('float32'),position(coords[b],graph['shape'],1)])
            ca=torch.tensor(coords[a,1:],dtype=torch.float32);cb=torch.tensor(coords[b,1:],dtype=torch.float32)
            f=model(fa,fb,ca,cb);r=model(fb,fa,cb,ca).T
            scale=(f.std(dim=0,unbiased=False).clamp_min(1e-4)/r.std(dim=0,unbiased=False).clamp_min(1e-4)).clamp(.5,2.)
            r=(r-r.mean(dim=0))*scale+f.mean(dim=0)
            p=1/(.85/f.softmax(dim=0).clamp_min(1e-8)+.15/r.softmax(dim=0).clamp_min(1e-8))
            p=(p/p.sum(dim=0).clamp_min(1e-8)).numpy()
            distance=np.linalg.norm((coords[a,None,1:]-coords[None,b,1:])*SCALE,axis=2)
            ii,jj=np.nonzero(distance<=20.)
            output.update({(ids[a[i]],ids[b[j]]):float(p[i,j]) for i,j in zip(ii,jj)})
            frames+=1
            if frames%20==0:print('REPLAY_FRAME',int(t),len(output),flush=True)
    return output,dict(mapped_nodes=len(mapping),reference_nodes=len(ids),mean_mapping_um=float(np.mean(distances)) if distances else None,frame_pairs=frames,visual_candidates=len(output))
