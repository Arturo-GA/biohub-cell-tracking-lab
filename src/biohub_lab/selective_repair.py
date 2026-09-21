"""Atomic disagreement-component gating, using frozen observable evidence only."""
import math
import numpy as np
SCALE=np.array([1.625,.40625,.40625])

def components(old,new):
    diff=set(old)^set(new);parent={}
    def root(k):
        parent.setdefault(k,k)
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for a,b in sorted(diff):
        x,y=root(('out',a)),root(('in',b));parent[x]=y
    groups={}
    for e in sorted(diff):groups.setdefault(root(('out',e[0])),set()).add(e)
    return list(groups.values())

def select(nodes,original,candidate,visual,mode):
    assert mode in ('confidence','trajectory','joint')
    old,new=set(map(tuple,original)),set(map(tuple,candidate));incoming={};outgoing={}
    for a,b in old:incoming.setdefault(b,[]).append(a);outgoing.setdefault(a,[]).append(b)
    xyz={k:np.array([v[a] for a in ('z','y','x')])*SCALE for k,v in nodes.items()}
    def curvature(e):
        a,b=e;velocity=xyz[b]-xyz[a];cost=[]
        prev=incoming.get(a,[]);nxt=outgoing.get(b,[])
        if len(prev)==1 and len(outgoing[prev[0]])==1 and nodes[a]['t']==nodes[prev[0]]['t']+1:
            cost.append(min(float(np.linalg.norm(velocity-(xyz[a]-xyz[prev[0]]))),10.))
        if len(nxt)==1 and len(incoming[nxt[0]])==1 and nodes[nxt[0]]['t']==nodes[b]['t']+1:
            cost.append(min(float(np.linalg.norm((xyz[nxt[0]]-xyz[b])-velocity)),10.))
        return float(np.mean(cost)) if cost else None
    def evidence(edges):return sum(math.log(max(1e-6,min(1.,visual.get(e,.05)))/.05) for e in edges)
    result=set(old);reports=[]
    for group in components(old,new):
        removed=group&old;added=group&new
        gain=evidence(added)-evidence(removed);size=max(1,len(added),len(removed))
        before=[curvature(e) for e in removed];after=[curvature(e) for e in added]
        covered=bool(before and after) and all(v is not None for v in before+after)
        before_mean=float(np.mean(before)) if covered else None;after_mean=float(np.mean(after)) if covered else None
        confident=gain>=math.log(2)*size
        coherent=covered and gain>1e-7 and after_mean<=before_mean+.5 and after_mean<=4.
        accept=confident if mode=='confidence' else coherent if mode=='trajectory' else confident and coherent
        if accept:result.difference_update(removed);result.update(added)
        reports.append(dict(accepted=bool(accept),removed=len(removed),added=len(added),log_gain=gain,before_curvature=before_mean,after_curvature=after_mean))
    # Changes are complete connected components: never half of an assignment swap.
    return sorted(result),dict(mode=mode,components=len(reports),accepted=sum(r['accepted'] for r in reports),removed=len(old-result),added=len(result-old),decisions=reports)
