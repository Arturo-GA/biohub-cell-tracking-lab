"""Add confirmed detected trajectories, preserving every original node and edge."""
import numpy as np
from scipy.spatial import cKDTree

SCALE=np.array([1.625,.40625,.40625])

def chains(coords,features):
    xyz=coords[:,1:]*SCALE;frames={int(t):np.flatnonzero(coords[:,0]==t) for t in np.unique(coords[:,0])}
    forward={};backward={}
    for t,a in frames.items():
        b=frames.get(t+1)
        if b is None or not len(a) or not len(b):continue
        d,j=cKDTree(xyz[b]).query(xyz[a],k=2);rd,rj=cKDTree(xyz[a]).query(xyz[b],k=2)
        for i in range(len(a)):
            k=int(j[i,0])
            if d[i,0]>6 or rj[k,0]!=i or d[i,1]-d[i,0]<.75 or rd[k,1]-rd[k,0]<.75:continue
            if float(np.dot(features[a[i]],features[b[k]]))<.75:continue
            forward[int(a[i])]=int(b[k]);backward[int(b[k])]=int(a[i])
    paths=[]
    for a in sorted(set(forward)-set(backward)):
        path=[a]
        while path[-1] in forward:path.append(forward[path[-1]])
        if len(path)>=5:paths.append(path)
    return paths

def augment(nodes,edges,coords,prob,features,standalone=False,precomputed_paths=None,original_mapping=None):
    coords=np.asarray(coords);prob=np.asarray(prob);features=np.asarray(features,np.float32)
    assert coords.shape==(len(prob),4) and len(features)==len(coords)
    assert np.isfinite(coords).all() and np.isfinite(prob).all() and np.isfinite(features).all()
    ids=sorted(nodes);original=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);mapped=np.full(len(coords),-1,np.int64)
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero(original[:,0]==t)
        if not len(b):continue
        dist,index=cKDTree(original[b,1:]*SCALE).query(coords[a,1:]*SCALE,k=2)
        nearest={}
        for i,k in enumerate(a):
            if dist[i,0]>3:continue
            mapped[k]=-2 # A nearby ambiguous baseline node is not an absent cell.
            if dist[i,1]-dist[i,0]<.75:continue
            target=ids[b[index[i,0]]]
            if target not in nearest or dist[i,0]<nearest[target][0]:nearest[target]=(dist[i,0],int(k))
        for target,(_,k) in nearest.items():mapped[k]=target
    if original_mapping is not None:
        mapped=np.asarray(original_mapping,np.int64).copy();assert len(mapped)==len(coords)
        assert all(k==-1 or k in nodes and np.array_equal(coords[i],[nodes[k][a] for a in ('t','z','y','x')]) for i,k in enumerate(mapped))
    paths=chains(coords,features) if precomputed_paths is None else precomputed_paths;proposals=[]
    for path in paths:
        j=0
        while j<len(path):
            if mapped[path[j]]!=-1:j+=1;continue
            begin=j
            while j<len(path) and mapped[path[j]]==-1:j+=1
            end=j;left=path[begin-1] if begin else None;right=path[end] if end<len(path) else None
            if left is not None and mapped[left]<0 or right is not None and mapped[right]<0:continue
            anchors=int(left is not None)+int(right is not None);fresh=path[begin:end]
            if anchors==0 and (not standalone or len(fresh)<8):continue
            if anchors==1 and len(fresh)<3:continue
            whole=([left] if left is not None else [])+fresh+([right] if right is not None else [])
            confidence=float(np.median(prob[fresh]));minimum=float(np.min(prob[fresh]))
            if confidence<(.8 if anchors==0 else .5) or minimum<(.5 if anchors==0 else .2):continue
            xyz=coords[whole,1:]*SCALE;acceleration=np.linalg.norm(np.diff(xyz,n=2,axis=0),axis=1)
            if len(acceleration) and np.max(acceleration)>3:continue
            score=(1+anchors)*confidence/(1+float(acceleration.mean()) if len(acceleration) else 1)
            proposals.append(dict(whole=whole,fresh=fresh,score=score,anchors=anchors))
    result={k:dict(v) for k,v in nodes.items()};links=set(map(tuple,edges));incoming={b for a,b in links};outgoing={a for a,b in links}
    cap=max(1,int(.01*len(nodes)));added=0;accepted=[];next_id=max(ids)+1
    for p in sorted(proposals,key=lambda p:(-p['score'],p['whole'][0])):
        if added+len(p['fresh'])>cap:continue
        first,last=p['whole'][0],p['whole'][-1]
        if mapped[first]>=0 and mapped[first] in outgoing or mapped[last]>=0 and mapped[last] in incoming:continue
        chain=[]
        for i in p['whole']:
            if mapped[i]>=0:chain.append(int(mapped[i]))
            else:
                result[next_id]=dict(zip(('t','z','y','x'),map(int,coords[i])));chain.append(next_id);next_id+=1
        new=list(zip(chain[:-1],chain[1:]));assert not any(a in outgoing or b in incoming for a,b in new)
        links.update(new);outgoing.update(a for a,b in new);incoming.update(b for a,b in new);added+=len(p['fresh']);accepted.append(p)
    return result,sorted(links),dict(raw_candidates=len(coords),persistent_chains=len(paths),proposals=len(proposals),accepted=len(accepted),
        anchored=sum(p['anchors']>0 for p in accepted),standalone=sum(p['anchors']==0 for p in accepted),added_nodes=added,added_edges=len(links)-len(edges),node_budget=cap)
