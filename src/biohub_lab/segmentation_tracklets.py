"""Frozen label-free graph proposal from temporally consistent new nuclei."""
import numpy as np
from scipy.spatial import cKDTree
from .nucverse_instances import SCALE
CONFIG=dict(exclusion_um=7.,link_um=5.,minimum_frames=2)
def recover(reference,reference_edges,frames,shape):
    nodes={k:dict(v) for k,v in reference.items()};edges=set(reference_edges);points={};candidate={};index=0
    for t,values in sorted(frames.items()):
        xyz=np.unique(np.rint(values).astype(int).reshape(-1,3),axis=0)
        assert 0<=t<shape[0] and np.all(xyz>=0) and np.all(xyz<np.array(shape[1:]))
        old=np.array([[v[a] for a in ('z','y','x')] for v in reference.values() if v['t']==t],float).reshape(-1,3)
        if len(old) and len(xyz):xyz=xyz[cKDTree(old*SCALE).query(xyz*SCALE)[0]>CONFIG['exclusion_um']]
        points[t]=[]
        for p in xyz:candidate[index]=(t,p);points[t].append(index);index+=1
    child={};parent={}
    for t,ids in points.items():
        other=points.get(t+1,[])
        if not ids or not other:continue
        a=np.array([candidate[k][1] for k in ids])*SCALE;b=np.array([candidate[k][1] for k in other])*SCALE
        distance,forward=cKDTree(b).query(a);_,backward=cKDTree(a).query(b)
        for i,k in enumerate(ids):
            if distance[i]<=CONFIG['link_um'] and backward[forward[i]]==i:
                target=other[forward[i]];child[k]=target;parent[target]=k
    next_id=max(nodes,default=-1)+1;added_nodes=added_edges=chains=0
    for first in sorted(set(candidate)-set(parent)):
        chain=[first]
        while chain[-1] in child:chain.append(child[chain[-1]])
        if len(chain)<CONFIG['minimum_frames']:continue
        ids=[];chains+=1
        for k in chain:
            t,p=candidate[k];nodes[next_id]=dict(t=int(t),z=int(p[0]),y=int(p[1]),x=int(p[2]));ids.append(next_id);next_id+=1
        edges.update(zip(ids[:-1],ids[1:]));added_nodes+=len(ids);added_edges+=len(ids)-1
    assert all(nodes[k]==v for k,v in reference.items()) and set(reference_edges)<=edges
    return nodes,sorted(edges),dict(config=CONFIG,unmatched_candidates=len(candidate),chains=chains,added_nodes=added_nodes,added_edges=added_edges,annotations_read=False)
