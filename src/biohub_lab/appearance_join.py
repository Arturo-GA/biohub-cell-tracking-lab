"""Join fragmented tracks using native appearance and two-sided extrapolation."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from .appearance_swap import describe,SCALE

ARMS={
 'strict':dict(radius=8.,residual=3.,appearance=.5,margin=.2),
 'balanced':dict(radius=12.,residual=5.,appearance=1.,margin=.15),
 'recall':dict(radius=14.,residual=8.,appearance=1.5,margin=.1),
}

def refine(nodes,edges,features,radius,residual,appearance,margin):
    report=dict(candidate_links=0,added_edges=0,changed_edges=0,nodes_changed=False,original_edges_removed=0)
    if not edges:return list(edges),report
    ids=sorted(nodes);ix={k:i for i,k in enumerate(ids)};coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);xyz=coords[:,1:]*SCALE
    pairs=np.array([(ix[a],ix[b]) for a,b in edges],int);inc=Counter(pairs[:,1]);out=Counter(pairs[:,0])
    parent={int(b):int(a) for a,b in pairs if out[a]==1 and inc[b]==1};child={int(a):int(b) for a,b in pairs if out[a]==1 and inc[b]==1}
    ends=[k for k in range(len(ids)) if not out[k] and k in parent and parent[k] in parent]
    starts=[k for k in range(len(ids)) if not inc[k] and k in child and child[k] in child]
    candidates=[]
    for t in np.unique(coords[:,0]):
        aa=np.array([k for k in ends if coords[k,0]==t],int);bb=np.array([k for k in starts if coords[k,0]==t+1],int)
        if not len(aa) or not len(bb):continue
        tree=cKDTree(xyz[bb]);neighbors=tree.query_ball_point(xyz[aa],radius)
        for a,js in zip(aa,neighbors):
            left=[parent[parent[a]],parent[a],a];va=(xyz[a]-xyz[left[0]])/2;fa=np.median(features[left],axis=0)
            for j in js:
                b=bb[j];right=[b,child[b],child[child[b]]];vb=(xyz[right[-1]]-xyz[b])/2
                motion=max(np.linalg.norm(xyz[b]-xyz[a]-va),np.linalg.norm(xyz[b]-xyz[a]-vb))
                if motion>residual:continue
                app=float(np.sum((fa-np.median(features[right],axis=0))**2))
                if app>appearance:continue
                cost=motion/residual+app/appearance+np.linalg.norm(xyz[b]-xyz[a])/radius
                candidates.append((float(cost),int(a),int(b)))
    report['candidate_links']=len(candidates);by_a={};by_b={}
    for cost,a,b in candidates:by_a.setdefault(a,[]).append((cost,b));by_b.setdefault(b,[]).append((cost,a))
    for d in (by_a,by_b):
        for v in d.values():v.sort()
    accepted=[]
    for cost,a,b in sorted(candidates):
        left,right=by_a[a],by_b[b]
        if left[0][1]!=b or right[0][1]!=a:continue
        if len(left)>1 and left[1][0]-cost<margin:continue
        if len(right)>1 and right[1][0]-cost<margin:continue
        accepted.append((ids[a],ids[b]))
    result=list(edges)+accepted;assert len(result)==len(set(result))
    assert all(out[ix[a]]==0 and inc[ix[b]]==0 for a,b in accepted)
    report.update(added_edges=len(accepted),changed_edges=len(accepted))
    return result,report
