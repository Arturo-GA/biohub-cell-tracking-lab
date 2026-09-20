"""Constrained lineage event hypotheses with perturb-and-MAP uncertainty.

Own implementation, not a reproduction or calibrated Bayesian posterior.
Keep detected nodes and incoming-edge coverage; allow forks to appear/disappear.
"""
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
SCALE=np.array([1.625,.40625,.40625])

def solve_events(events,cost,nparents,nchildren,forks,limit=3.):
    rows=[];cols=[];values=[]
    for k,(parent,kids) in enumerate(events):
        rows.append(parent);cols.append(k);values.append(1)
        for child in kids:rows.append(nparents+child);cols.append(k);values.append(1)
        if len(kids)==2:rows.append(nparents+nchildren);cols.append(k);values.append(1)
    a=coo_matrix((values,(rows,cols)),shape=(nparents+nchildren+1,len(events))).tocsc()
    lower=np.r_[np.zeros(nparents),np.ones(nchildren),0 if forks is None else forks];upper=np.r_[np.ones(nparents),np.ones(nchildren),nchildren//2 if forks is None else forks]
    r=milp(np.asarray(cost,float),integrality=np.ones(len(events)),bounds=Bounds(0,1),constraints=LinearConstraint(a,lower,upper),options={'time_limit':limit,'mip_rel_gap':0.})
    if r.status!=0 or r.x is None:return None
    chosen=np.flatnonzero(r.x>.5);counts=Counter(c for k in chosen for c in events[k][1]);assert counts==Counter(range(nchildren))
    assert len({events[k][0] for k in chosen})==len(chosen)
    if forks is not None:assert sum(len(events[k][1])==2 for k in chosen)==forks
    return chosen

def reassign_uncertain(ids,coords,edges,features,seed=430920):
    idx={int(n):i for i,n in enumerate(ids)};pos=coords[:,1:]*SCALE;edges=[tuple(map(int,e)) for e in edges];children={};parent={}
    for a,b in edges:children.setdefault(a,[]).append(b);parent[b]=a
    rng=np.random.default_rng(seed);answers={'map':[],'consensus':[]};reports=[]
    rate=np.clip(sum(len(v)==2 for v in children.values())/max(len(children),1),1e-4,.2)
    new_division_penalty=float(np.log((1-rate)/rate))
    for t in sorted(set(int(coords[idx[b],0]) for a,b in edges)):
        original=[(a,b) for a,b in edges if int(coords[idx[b],0])==t]
        old_parents=set(a for a,b in original)
        ps=sorted(int(n) for n,c in zip(ids,coords) if int(c[0])==t-1);cs=sorted(set(b for a,b in original));pi={n:i for i,n in enumerate(ps)};ci={n:i for i,n in enumerate(cs)}
        assert all(coords[idx[a],0]==t-1 for a in ps)
        pp=pos[[idx[n] for n in ps]];cp=pos[[idx[n] for n in cs]];velocity=np.zeros_like(pp)
        for i,n in enumerate(ps):
            if n in parent:velocity[i]=pp[i]-pos[idx[parent[n]]]
        neigh=cKDTree(pp).query(pp,k=min(12,len(pp)))[1];neigh=np.asarray(neigh).reshape(len(pp),-1)
        local=np.median(velocity[neigh],axis=1);variance=np.maximum(np.var(velocity[neigh]-local[:,None,:],axis=1),1.)
        mean=pp+.5*velocity+.5*local;distance,near=cKDTree(cp).query(mean,k=min(6,len(cp)),distance_upper_bound=12.)
        distance=np.asarray(distance).reshape(len(pp),-1);near=np.asarray(near).reshape(len(pp),-1)
        events=[];cost=[];old_indices=[];forks=sum(len(children.get(n,[]))==2 for n in ps)
        active_rate=np.clip(len(old_parents)/max(len(ps),1),.5,.995)
        new_parent_penalty=float(np.log(active_rate/(1-active_rate)))
        h=features.astype(np.float32);h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-8)
        for i,n in enumerate(ps):
            old=tuple(sorted(ci[c] for c in children.get(n,[]) if c in ci));candidates={int(k) for k,d in zip(near[i],distance[i]) if np.isfinite(d)};candidates.update(old)
            options=[(k,) for k in sorted(candidates)]+list(combinations(sorted(candidates),2))
            for kids in options:
                if len(kids)==2 and np.linalg.norm(cp[kids[0]]-cp[kids[1]])>12 and kids!=old:continue
                target=cp[list(kids)].mean(0);delta=target-mean[i]
                motion=float(np.sum(delta**2/variance[i])/2)
                appearance=float(np.mean([1-np.dot(h[idx[n]],h[idx[cs[k]]]) for k in kids]))
                # Look ahead via existing daughter continuation, without annotated trajectories.
                future=[]
                for k in kids:
                    nxt=children.get(cs[k],[])
                    if len(nxt)==1:future.append(pos[idx[nxt[0]]]-cp[k])
                kinetic=float(np.sum((np.mean(future,0)-(target-pp[i]))**2/variance[i])/2) if future else 0.
                value=motion+.25*kinetic+3*appearance-1.*(kids==old)
                if len(kids)==2 and len(old)!=2:value+=new_division_penalty
                if not old:value+=new_parent_penalty
                if kids==old:old_indices.append(len(events))
                events.append((i,kids));cost.append(value)
        assert len(old_indices)==len(old_parents)
        chosen=solve_events(events,cost,len(ps),len(cs),None);optimal=chosen is not None
        if chosen is None:chosen=np.array(old_indices)
        frequency=np.zeros(len(events));sample_optimal=0
        for _ in range(3):
            sample=solve_events(events,np.asarray(cost)-.5*rng.gumbel(size=len(events)),len(ps),len(cs),None)
            if sample is None:sample=chosen
            else:sample_optimal+=1
            frequency[sample]+=1/3
        null_frequency=np.ones(len(ps))
        for k,(i,kids) in enumerate(events):null_frequency[i]-=frequency[k]
        null_frequency=np.clip(null_frequency,0,1)
        # Include the unselected-parent state. Raw frequency sums would favor
        # more singleton events merely because their event count is larger.
        consensus_cost=np.array([-np.log((frequency[k]+.1)/(null_frequency[i]+.1)) for k,(i,kids) in enumerate(events)])+.001*np.asarray(cost)
        consensus=solve_events(events,consensus_cost,len(ps),len(cs),None)
        consensus_optimal=consensus is not None
        if consensus is None:consensus=chosen
        for arm,selection in [('map',chosen),('consensus',consensus)]:answers[arm].extend((ps[events[k][0]],cs[c]) for k in selection for c in events[k][1])
        reports.append(dict(frame=t,events=len(events),original_forks=forks,map_forks=sum(len(events[k][1])==2 for k in chosen),consensus_forks=sum(len(events[k][1])==2 for k in consensus),map_optimal=optimal,consensus_optimal=consensus_optimal,sample_optimal=sample_optimal,uncertain_events=int(np.sum((frequency>0)&(frequency<1)))))
        if t%20==0:print('UNCERT_FRAME',reports[-1],flush=True)
    incoming=Counter(b for a,b in edges)
    for arm,e in answers.items():
        assert Counter(b for a,b in e)==incoming and len(e)==len(edges) and len(set(e))==len(e)
        assert max(Counter(a for a,b in e).values(),default=0)<=2
    return answers,dict(frames=reports,new_division_penalty=new_division_penalty,changed_edges={a:len(set(e)-set(edges)) for a,e in answers.items()})
