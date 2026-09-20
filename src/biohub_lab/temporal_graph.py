"""Reassign continuation parents while preserving every node degree and division."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
SCALE=np.array([1.625,.40625,.40625])

def reassign(ids,coords,edges,features,image=True,prior=1.):
    index={int(n):i for i,n in enumerate(ids)};edges=[tuple(map(int,e)) for e in edges]
    degree=Counter(a for a,b in edges);incoming=Counter(b for a,b in edges);groups={};locked=[]
    for a,b in edges:
        ia,ib=index[a],index[b]
        if degree[a]==1 and incoming[b]==1 and coords[ib,0]==coords[ia,0]+1:groups.setdefault(int(coords[ib,0]),[]).append((a,b))
        else:locked.append((a,b))
    answer=list(locked)
    for t,original in groups.items():
        a=sorted(a for a,b in original);b=sorted(b for a,b in original);ai={n:i for i,n in enumerate(a)};old={child:parent for parent,child in original}
        apos=coords[[index[n] for n in a],1:]*SCALE;bpos=coords[[index[n] for n in b],1:]*SCALE
        distance,neighbor=cKDTree(apos).query(bpos,k=min(12,len(a)),distance_upper_bound=20.)
        distance=np.asarray(distance).reshape(len(b),-1);neighbor=np.asarray(neighbor).reshape(len(b),-1)
        row=[];col=[];cost=[]
        for j,child in enumerate(b):
            candidates={int(i) for i,d in zip(neighbor[j],distance[j]) if np.isfinite(d)};candidates.add(ai[old[child]])
            for i in sorted(candidates):
                parent=a[i];d=float(np.linalg.norm(apos[i]-bpos[j]));cos=float(np.dot(features[index[parent]],features[index[child]])) if image else 0.
                # Positive constant does not change a complete assignment objective.
                value=100.+d/5.-cos/.2-prior*(old[child]==parent)
                row.append(j);col.append(i);cost.append(value)
        matrix=coo_matrix((cost,(row,col)),shape=(len(b),len(a))).tocsr();rr,cc=min_weight_full_bipartite_matching(matrix)
        answer.extend((a[int(i)],b[int(j)]) for j,i in zip(rr,cc))
    assert len(set(answer))==len(edges) and Counter(a for a,b in answer)==degree and Counter(b for a,b in answer)==incoming
    assert set(locked)<=set(answer)
    return sorted(answer),dict(changed_edges=len(set(answer)-set(edges)),locked_edges=len(locked))
