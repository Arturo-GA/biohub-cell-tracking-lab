"""Second-order association using frozen Harmonic visual probabilities.

No labels or new detections. Divisions and links absent from the visual cache
are locked. Frame assignments are optimized against a whole-sequence energy.
"""
import numpy as np
from .association_rank import SCALE
from .detection_identity import maximum_vote_edges


def refine(nodes, original, visual, temporal_weight=.75):
    ids=sorted(nodes); index={k:i for i,k in enumerate(ids)}
    coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids],float)
    xyz=coords[:,1:]*SCALE
    original=set(map(tuple,original)); degree={}
    for a,b in original:degree[a]=degree.get(a,0)+1
    locked={e for e in original if degree[e[0]]>1 or e not in visual or nodes[e[1]]['t']!=nodes[e[0]]['t']+1}
    occupied_out={a for a,b in locked}; occupied_in={b for a,b in locked}
    allowed={e:float(p) for e,p in visual.items() if e[0] in index and e[1] in index
             and e[0] not in occupied_out and e[1] not in occupied_in
             and nodes[e[1]]['t']==nodes[e[0]]['t']+1 and np.isfinite(p)}
    pairs=sorted(allowed)
    if not pairs:return sorted(original),dict(alternatives=0,changed=0)
    edges=np.array([(index[a],index[b]) for a,b in pairs],np.int64)
    base=np.log(np.clip([allowed[e] for e in pairs],1e-6,1)/.05)
    lookup={tuple(e):i for i,e in enumerate(edges)}
    fixed=np.array([(index[a],index[b]) for a,b in sorted(locked)],np.int64).reshape(-1,2)
    current=np.array([(index[a],index[b]) for a,b in sorted(original-locked)],np.int64).reshape(-1,2)
    assert all(tuple(e) in lookup for e in current)
    def penalty(v,w):return np.minimum(np.linalg.norm(v-w,axis=-1)/3.,4.)
    def context(chosen):
        parent=np.full(len(ids),-1,int); child=parent.copy()
        for a,b in np.concatenate((fixed,chosen)):
            parent[b]=a
            if degree.get(ids[a],0)<=1:child[a]=b
        return parent,child
    def energy(chosen):
        selected=np.concatenate((fixed,chosen)); _,child=context(chosen)
        if not len(selected):return 0.
        a,b=selected.T; c=child[b]; valid=c>=0
        # Each consecutive edge pair is counted once, including fixed boundaries.
        value=sum(base[lookup[tuple(e)]]+.01/(1+np.linalg.norm(xyz[e[1]]-xyz[e[0]])) for e in chosen)
        return float(value-temporal_weight*penalty(xyz[b[valid]]-xyz[a[valid]],xyz[c[valid]]-xyz[b[valid]]).sum())
    history=[energy(current)]; times=np.unique(coords[edges[:,0],0])
    for sweep in range(6):
        changed=0
        for t in (times if sweep%2==0 else times[::-1]):
            mask=coords[edges[:,0],0]==t; candidates=edges[mask]; value=base[mask].copy()
            parent,child=context(current); a,b=candidates.T; p,q=parent[a],child[b]
            v=xyz[b]-xyz[a]; ok=p>=0
            value[ok]-=temporal_weight*penalty(v[ok],xyz[a[ok]]-xyz[p[ok]])
            ok=q>=0; value[ok]-=temporal_weight*penalty(v[ok],xyz[q[ok]]-xyz[b[ok]])
            selected=maximum_vote_edges(coords,candidates,value)
            proposal=np.concatenate((current[coords[current[:,0],0]!=t],selected))
            score=energy(proposal)
            if score>history[-1]+1e-7:
                current=proposal;history.append(score);changed+=1
        if not changed:break
    result=locked|{(ids[a],ids[b]) for a,b in current}
    assert locked<=result
    return sorted(result),dict(visual_candidates=len(pairs),alternatives=len(set(pairs)-original),locked=len(locked),
        removed=len(original-result),added=len(result-original),initial_energy=history[0],final_energy=history[-1],
        accepted_blocks=len(history)-1,temporal_weight=temporal_weight)
