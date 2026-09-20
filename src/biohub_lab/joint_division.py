"""Counterfactual division events with displaced parents and anchored detections."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
SCALE=np.array([1.625,.40625,.40625])

def tables(edges):
    prev=defaultdict(list);nxt=defaultdict(list)
    for a,b in edges:prev[int(b)].append(int(a));nxt[int(a)].append(int(b))
    return prev,nxt

def tail(n,coords,nxt):
    chain=[int(n)]
    for _ in range(2):
        future=nxt[chain[-1]]
        if len(future)!=1 or coords[future[0],0]!=coords[chain[-1],0]+1:return None
        chain.append(future[0])
    return chain

def proposals(coords,edges,dense):
    """No annotations; every eligible mother/frame is searched with fixed rules."""
    coords=np.asarray(coords);xyz=coords[:,1:]*SCALE;dxyz=dense[:,1:]*SCALE
    prev,nxt=tables(edges);old=set(map(tuple,edges));chains={n:tail(n,coords,nxt) for n in range(len(coords))}
    velocity={n:xyz[n]-xyz[prev[n][0]] for n in range(len(coords)) if len(prev[n])==1}
    residual=[np.linalg.norm(xyz[b]-xyz[a]-velocity[a]) for a,b in edges if a in velocity and len(nxt[int(a)])==1]
    sigma=float(np.clip(np.median(residual) if residual else 2.,1.,4.))
    def edgecost(a,b):return min(float(np.sum((xyz[b]-xyz[a]-velocity.get(a,np.zeros(3)))**2)/sigma**2),16.)
    donors=defaultdict(list)
    # Reconstruct one missing frame immediately before an existing two-frame tail.
    for t in np.unique(coords[:,0]).astype(int):
        hi=np.flatnonzero(coords[:,0]==t);di=np.flatnonzero(dense[:,0]==t)
        if not len(di):continue
        if len(hi):di=di[cKDTree(xyz[hi]).query(dxyz[di])[0]>3.]
        if not len(di):continue
        tree=cKDTree(dxyz[di])
        for anchor in np.flatnonzero(coords[:,0]==t+1):
            future=nxt[int(anchor)]
            if len(future)!=1 or coords[future[0],0]!=t+2:continue
            if prev[int(anchor)] and len(nxt[prev[int(anchor)][0]])!=1:continue
            expected=2*xyz[anchor]-xyz[future[0]]
            distance,index=tree.query(expected,k=min(2,len(di)),distance_upper_bound=4.)
            distance=np.atleast_1d(distance);index=np.atleast_1d(index)
            if not np.isfinite(distance[0]):continue
            if len(distance)>1 and distance[1]-distance[0]<1.:continue
            d=int(di[index[0]])
            if np.linalg.norm(dxyz[d]-xyz[anchor])>8:continue
            donors[t].append(dict(dense=d,anchor=int(anchor),future=int(future[0]),residual=float(distance[0])))
    donor_trees={t:cKDTree(np.asarray([dxyz[d['dense']] for d in values])) for t,values in donors.items() if values}
    events=[];eligible=0
    for t in np.unique(coords[:,0]).astype(int):
        child_indices=np.array([n for n in np.flatnonzero(coords[:,0]==t+1) if chains[int(n)] is not None],dtype=int)
        tree=cKDTree(xyz[child_indices]) if len(child_indices) else None
        for mother in np.flatnonzero(coords[:,0]==t):
            mother=int(mother)
            if len(prev[mother])!=1 or len(nxt[mother])!=1:continue
            first=nxt[mother][0];first_tail=chains[first]
            if first_tail is None:continue
            eligible+=1;possibilities=[]
            if tree is not None:
                dd,ii=tree.query(xyz[mother],k=min(5,len(child_indices)),distance_upper_bound=20.)
                for distance,index in zip(np.atleast_1d(dd),np.atleast_1d(ii)):
                    if not np.isfinite(distance):continue
                    second=int(child_indices[int(index)])
                    if second==first:continue
                    p=prev[second]
                    if p and (p[0]==mother or len(nxt[p[0]])!=1):continue
                    possibilities.append(dict(kind='parent',second=second,chain=chains[second]))
            if t+1 in donor_trees:
                for di in sorted(donor_trees[t+1].query_ball_point(xyz[mother],20.)):
                    possibilities.append(dict(kind='donor',**donors[t+1][di]))
            for option in possibilities:
                A=xyz[first_tail]
                if option['kind']=='parent':
                    second=option['second'];B=xyz[option['chain']]
                    removed=[(prev[second][0],second)] if prev[second] else []
                    additions=[(mother,second)];new_dense=[];extra=0.
                else:
                    d,anchor,future=option['dense'],option['anchor'],option['future'];second=len(coords)+d
                    B=np.vstack([dxyz[d],xyz[anchor],xyz[future]])
                    removed=[(prev[anchor][0],anchor)] if prev[anchor] else []
                    additions=[(mother,second),(second,anchor)];new_dense=[d]
                    extra=2.+min(option['residual']**2/sigma**2,16.)
                if set(first_tail)&set(option.get('chain',[option.get('anchor'),option.get('future')])):continue
                if any(a==mother or b==first for a,b in removed):continue
                prediction=xyz[mother]+velocity[mother]
                u,v=A[0]-prediction,B[0]-prediction
                denom=np.linalg.norm(u)*np.linalg.norm(v)
                if denom<1e-6 or np.dot(u,v)/denom>-.25:continue
                sep=np.linalg.norm(A-B,axis=1)
                if not 2.<=sep[0]<=16. or sep[-1]<sep[0]+.5:continue
                center=(A[0]+B[0])/2
                com_velocity=((A[1:]+B[1:])-(A[:-1]+B[:-1]))/2
                split=float(np.sum((center-prediction)**2)/sigma**2)
                split+=float(np.mean(np.sum((com_velocity-velocity[mother])**2,axis=1))/sigma**2)
                baseline=edgecost(mother,first)+sum(edgecost(a,b) for a,b in removed)
                # Fixed dimensionless split, termination and missing-detection priors.
                proposed=split+3.+4.*len(removed)+extra
                gain=baseline-proposed
                resources={mother,first,second,*[n for e in removed+additions for n in e]}
                resources.update(first_tail);resources.update(option.get('chain',[option.get('anchor'),option.get('future')]))
                resources.discard(None)
                events.append(dict(mother=mother,kind=option['kind'],remove=removed,add=additions,dense=new_dense,gain=gain,resources=sorted(resources),baseline_cost=baseline,proposed_cost=proposed))
    return events,dict(sigma_um=sigma,eligible_mothers=eligible,anchored_donor_hypotheses=sum(map(len,donors.values())),events=len(events),positive_gain=sum(e['gain']>1 for e in events))

def select(coords,edges,dense,events,joint):
    pool=[e for e in events if e['gain']>1. and (joint or e['kind']=='parent')]
    selected=[];solver=None
    if pool:
        keys=sorted({k for e in pool for k in e['resources']});index={k:i for i,k in enumerate(keys)};rr=[];cc=[]
        for j,e in enumerate(pool):
            for k in e['resources']:rr.append(index[k]);cc.append(j)
        matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(keys),len(pool))).tocsc()
        result=milp(-np.array([e['gain'] for e in pool]),integrality=np.ones(len(pool)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,0,1),options=dict(time_limit=60,mip_rel_gap=0.))
        if result.x is None:raise RuntimeError('Joint event optimization produced no feasible assignment')
        selected=[e for e,x in zip(pool,result.x) if x>.5]
        used=[]
        for e in selected:used.extend(e['resources'])
        if len(used)!=len(set(used)):raise RuntimeError('Conflicting optimizer result')
        solver=dict(status=int(result.status),message=str(result.message),gap=float(result.mip_gap))
    answer=set(map(tuple,edges));donor_ids=sorted({d for e in selected for d in e['dense']})
    remap={len(coords)+d:len(coords)+i for i,d in enumerate(donor_ids)}
    for e in selected:
        assert set(map(tuple,e['remove']))<=answer
        answer.difference_update(map(tuple,e['remove']));answer.update((remap.get(a,a),remap.get(b,b)) for a,b in e['add'])
    new_coords=np.vstack([coords,dense[donor_ids]]) if donor_ids else coords.copy()
    new_edges=np.asarray(sorted(answer),np.int64).reshape(-1,2);prev,nxt=tables(new_edges)
    assert all(len(v)<=1 for v in prev.values()) and all(len(v)<=2 for v in nxt.values())
    assert all(new_coords[b,0]==new_coords[a,0]+1 for a,b in new_edges)
    return new_coords,new_edges,dict(selected=len(selected),parent_events=sum(e['kind']=='parent' for e in selected),donor_events=sum(e['kind']=='donor' for e in selected),added_nodes=len(donor_ids),removed_edges=sum(len(e['remove']) for e in selected),added_edges=sum(len(e['add']) for e in selected),solver=solver),selected
