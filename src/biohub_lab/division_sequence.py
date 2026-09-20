"""Symmetric, multi-frame mother/daughter features and conservative graph edits."""
from itertools import combinations
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, gaussian_laplace, map_coordinates

SCALE=np.array([1.625,.40625,.40625])

def links(edges):
    previous=defaultdict(list); following=defaultdict(list)
    for a,b in edges: following[int(a)].append(int(b)); previous[int(b)].append(int(a))
    return previous,following

def context(triple,coords,previous,following):
    m,a,b=map(int,triple); before=[m]
    for _ in range(2):
        pp=previous[before[-1]]
        if len(pp)!=1 or coords[before[-1],0]-coords[pp[0],0]!=1: return None
        before.append(pp[0])
    chains=[]
    for child in [a,b]:
        chain=[child]
        for _ in range(2):
            ff=following[chain[-1]]
            if len(ff)!=1 or coords[ff[0],0]-coords[chain[-1],0]!=1: return None
            chain.append(ff[0])
        chains.append(chain)
    if set(chains[0])&set(chains[1]): return None
    return list(reversed(before))+chains[0]+chains[1]

def candidates(coords,edges,predicted=False):
    prev,nxt=links(edges); xyz=coords[:,1:]*SCALE; triples=[]; contexts=[]
    for t in np.unique(coords[:,0]):
        mothers=np.flatnonzero(coords[:,0]==t); daughters=np.flatnonzero(coords[:,0]==t+1)
        if not len(daughters): continue
        tree=cKDTree(xyz[daughters])
        for m in mothers:
            existing=set(nxt[int(m)])
            close=daughters[tree.query_ball_point(xyz[m],20.)]
            if predicted: close=np.array([c for c in close if not prev[int(c)] or int(c) in existing],dtype=int)
            close=sorted(close,key=lambda c:float(np.linalg.norm(xyz[c]-xyz[m])))[:4]
            if predicted: close=sorted(set(close)|existing)
            for a,b in combinations(close,2):
                if predicted and existing and not existing.issubset({int(a),int(b)}): continue
                c=context((m,a,b),coords,prev,nxt)
                if c is not None: triples.append((m,a,b)); contexts.append(c)
    return np.asarray(triples,np.int64).reshape(-1,3),np.asarray(contexts,np.int64).reshape(-1,9)

def descriptors(volume,positions):
    volume=np.asarray(volume,np.float32)
    lo,hi=np.percentile(volume,[2,99.8]); v=np.clip((volume-lo)/max(float(hi-lo),1e-6),0,1)
    g1=gaussian_filter(v,1); g2=gaussian_filter(v,2)
    std=np.sqrt(np.maximum(gaussian_filter(v*v,1)-g1*g1,0))
    arrays=[v,g1,g2,std,gaussian_laplace(v,1),g1-g2]
    p=np.asarray(positions).T/np.array([1,4,4])[:,None]
    return np.stack([map_coordinates(a,p,order=1,mode='nearest') for a in arrays],axis=1).astype(np.float32)

def features(coords,h,contexts):
    if not len(contexts): return np.empty((0,22)),np.empty((0,18)),np.empty((0,54))
    c=coords[contexts,1:]*SCALE; f=h[contexts]
    m,a,b=c[:,:3],c[:,3:6],c[:,6:9]
    sep=np.linalg.norm(a-b,axis=2)
    da=np.linalg.norm(a-m[:,2,None],axis=2); db=np.linalg.norm(b-m[:,2,None],axis=2)
    ma=np.linalg.norm(np.diff(m,axis=1),axis=2)
    va=np.linalg.norm(np.diff(a,axis=1),axis=2); vb=np.linalg.norm(np.diff(b,axis=1),axis=2)
    midpoint=(a+b)/2
    com=np.linalg.norm(midpoint-m[:,2,None],axis=2)
    alignment=np.sum((a[:,0]-m[:,2])*(b[:,0]-m[:,2]),axis=1,keepdims=True)
    geometry=np.concatenate([sep,da+db,np.abs(da-db),ma,va+vb,np.abs(va-vb),com,alignment,sep[:,1:]-sep[:,:-1],np.linalg.norm(midpoint[:,0]-(m[:,2]+m[:,2]-m[:,1]),axis=1,keepdims=True)],axis=1)/10
    static=np.concatenate([f[:,2],(f[:,3]+f[:,6])/2,np.abs(f[:,3]-f[:,6])],axis=1)
    sequence=np.concatenate([f[:,:3].reshape(len(f),-1),((f[:,3:6]+f[:,6:9])/2).reshape(len(f),-1),np.abs(f[:,3:6]-f[:,6:9]).reshape(len(f),-1)],axis=1)
    return geometry.astype(np.float32),static.astype(np.float32),sequence.astype(np.float32)

def labels(triples,identity,truth_edges):
    parent={int(b):int(a) for a,b in truth_edges}; result=[]
    for m,a,b in triples:
        im,ia,ib=map(int,identity[[m,a,b]])
        pa,pb=parent.get(ia),parent.get(ib)
        if min(im,ia,ib)<0 or pa is None or pb is None: result.append(-1)
        else: result.append(int(pa==im and pb==im and ia!=ib))
    return np.asarray(result,np.int8)

def decode(edges,triples,scores,threshold,coords):
    answer=set(map(tuple,edges)); prev,nxt=links(edges); edits=[]; claimed=set()
    for i in np.argsort(-scores,kind='stable'):
        m,a,b=map(int,triples[i]); original=set(nxt[m]); target={a,b}
        if len(original)==2: continue  # Existing divisions stay untouched in this first recall test.
        if scores[i]<threshold or m in claimed or not original.issubset(target): continue
        if any(prev[c] and prev[c]!=[m] for c in target): continue
        if any(any(v==c for _,v in answer) and c not in original for c in target): continue
        for c in target-original: answer.add((m,c))
        claimed.add(m); edits.append(dict(mother=m,daughters=sorted(target),score=float(scores[i])))
    return np.asarray(sorted(answer),np.int64).reshape(-1,2),edits
