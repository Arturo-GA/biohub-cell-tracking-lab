"""Exact separable visual-only arm of E022, without repeated sequence scans."""
import numpy as np
from .association_rank import SCALE
from .detection_identity import maximum_vote_edges

def assign(nodes,original,visual):
    ids=sorted(nodes);index={k:i for i,k in enumerate(ids)}
    coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids],float)
    xyz=coords[:,1:]*SCALE;original=set(map(tuple,original));degree={}
    for a,b in original:degree[a]=degree.get(a,0)+1
    locked={e for e in original if degree[e[0]]>1 or e not in visual or nodes[e[1]]['t']!=nodes[e[0]]['t']+1}
    occupied_out={a for a,b in locked};occupied_in={b for a,b in locked}
    allowed={e:float(p) for e,p in visual.items() if e[0] in index and e[1] in index
        and e[0] not in occupied_out and e[1] not in occupied_in
        and nodes[e[1]]['t']==nodes[e[0]]['t']+1 and np.isfinite(p)}
    pairs=sorted(allowed)
    if not pairs:return sorted(original),dict(alternatives=0,added=0,removed=0)
    edges=np.array([(index[a],index[b]) for a,b in pairs],np.int64)
    base=np.log(np.clip([allowed[e] for e in pairs],1e-6,1)/.05)
    utility=base+.01/(1+np.linalg.norm(xyz[edges[:,1]]-xyz[edges[:,0]],axis=1))
    lookup={tuple(e):float(v) for e,v in zip(edges,utility)}
    current=np.array([(index[a],index[b]) for a,b in sorted(original-locked)],np.int64).reshape(-1,2)
    assert all(tuple(e) in lookup for e in current)
    chosen=maximum_vote_edges(coords,edges,base);output=[];before=after=0.;changed=0
    for t in np.unique(coords[edges[:,0],0]):
        old=current[coords[current[:,0],0]==t];new=chosen[coords[chosen[:,0],0]==t]
        previous=sum(lookup[tuple(e)] for e in old);proposed=sum(lookup[tuple(e)] for e in new)
        before+=previous
        if proposed>previous+1e-7:output.extend(map(tuple,new));after+=proposed;changed+=1
        else:output.extend(map(tuple,old));after+=previous
    result=locked|{(ids[a],ids[b]) for a,b in output}
    return sorted(result),dict(visual_candidates=len(pairs),alternatives=len(set(pairs)-original),locked=len(locked),
        removed=len(original-result),added=len(result-original),initial_energy=before,final_energy=after,
        accepted_blocks=changed,temporal_weight=0.,implementation='Separable assignment equivalent to E022 visual arm')
