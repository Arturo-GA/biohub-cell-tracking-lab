"""Robust full-track penalized acceleration smoothing with identity guards."""
from collections import Counter
import numpy as np
from scipy.linalg import solve_banded
from .native_centers import apply_offsets,SCALE

ARMS={'weak':dict(strength=.5,max_shift=1.2),'medium':dict(strength=2.,max_shift=1.2),'strong':dict(strength=8.,max_shift=1.2)}

def smooth_path(points,strength):
    n=len(points)
    if n<5:return points.copy()
    # D2.T @ D2 pentadiagonal bands; natural boundary conditions.
    main=np.full(n,6.);main[[0,-1]]=1.;main[[1,-2]]=5.
    adjacent=np.full(n-1,-4.);adjacent[[0,-1]]=-2.
    result=points.copy()
    for _ in range(4):
        residual=np.linalg.norm(result-points,axis=1);weights=1/np.sqrt(1+(residual/1.5)**2)
        bands=np.zeros((5,n));bands[0,2:]=strength;bands[1,1:]=strength*adjacent;bands[2]=weights+strength*main;bands[3,:-1]=strength*adjacent;bands[4,:-2]=strength
        result=solve_banded((2,2),bands,weights[:,None]*points,check_finite=False)
    return result

def refine(nodes,edges,strength,max_shift,shape):
    if not nodes:return {},dict(nodes=0,moved=0,mean_shift_um=0.,max_shift_um=0.,topology_changed=False)
    ids=sorted(nodes);ix={k:i for i,k in enumerate(ids)};coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);xyz=coords[:,1:]*SCALE
    inc=Counter(b for a,b in edges);out=Counter(a for a,b in edges);forward={a:b for a,b in edges if out[a]==1 and inc[b]==1};children=set(forward.values());delta=np.zeros_like(xyz,float);paths=0
    for start in sorted(set(forward)-children):
        chain=[start]
        while chain[-1] in forward:chain.append(forward[chain[-1]])
        if len(chain)<5:continue
        at=np.array([ix[k] for k in chain]);change=smooth_path(xyz[at],strength)-xyz[at];length=np.linalg.norm(change,axis=1);change*=np.minimum(1,max_shift/np.maximum(length,1e-12))[:,None];delta[at]=change/SCALE;paths+=1
    new,r=apply_offsets(nodes,edges,delta,1.,False,shape);r['smoothed_paths']=paths
    return new,r
