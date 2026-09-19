"""Collective-motion compensation and sequence-level trajectory assignment on CPU.

Coordinate descent over frame matchings with velocity consistency across up to
four consecutive edges. Two label-free initializations; no learned pair scores.
"""
import numpy as np
from scipy.spatial import cKDTree
from .association_rank import SCALE
from .detection_identity import maximum_vote_edges

CONFIG=dict(neighbors=16,seed_radius_um=8.,local_radius_um=30.,candidate_radius_um=20.,
            candidates=12,sweeps=4,adjacent_weight=.35,long_weight=.2)

def tissue_flow(coords):
    xyz=coords[:,1:].astype(float)*SCALE;flow=np.zeros_like(xyz);receipt=[]
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero(coords[:,0]==t+1)
        if not len(b):continue
        distance,forward=cKDTree(xyz[b]).query(xyz[a]);_,back=cKDTree(xyz[a]).query(xyz[b])
        ok=(back[forward]==np.arange(len(a)))&(distance<=CONFIG['seed_radius_um'])
        seeds=a[ok];vectors=xyz[b[forward[ok]]]-xyz[seeds]
        if len(seeds):
            k=min(CONFIG['neighbors'],len(seeds))
            dist,ix=cKDTree(xyz[seeds]).query(xyz[a],k=k)
            dist,ix=dist.reshape(len(a),k),ix.reshape(len(a),k)
            valid=dist<=CONFIG['local_radius_um'];local=np.where(valid[:,:,None],vectors[ix],np.nan)
            eligible=valid.any(1)
            flow[a]=np.median(vectors,axis=0)
            flow[a[eligible]]=np.nanmedian(local[eligible],axis=1)
        receipt.append(dict(frame=int(t),mutual_seeds=len(seeds),nodes=len(a)))
    return flow,receipt

def candidate_edges(coords,flow):
    xyz=coords[:,1:]*SCALE;parts=[]
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero(coords[:,0]==t+1)
        if not len(b):continue
        tree=cKDTree(xyz[b]);k=min(CONFIG['candidates'],len(b))
        # Union retains unregistered alternatives when flow seeds are wrong.
        for query in [xyz[a],xyz[a]+flow[a]]:
            d,ix=tree.query(query,k=k,distance_upper_bound=CONFIG['candidate_radius_um'])
            d,ix=d.reshape(len(a),k),ix.reshape(len(a),k)
            rows,cols=np.nonzero(np.isfinite(d));parts.append(np.c_[a[rows],b[ix[rows,cols]]])
    return np.unique(np.concatenate(parts),axis=0) if parts else np.empty((0,2),np.int64)

def links(n,edges):
    parent=np.full(n,-1,np.int64);child=parent.copy()
    if len(edges):parent[edges[:,1]]=edges[:,0];child[edges[:,0]]=edges[:,1]
    return parent,child

def penalty(a,b):return np.minimum(np.linalg.norm(a-b,axis=-1),12.)/3.

def edge_utilities(coords,flow,edges,current):
    xyz=coords[:,1:]*SCALE;parent,child=links(len(coords),current)
    a,b=edges.T;v=xyz[b]-xyz[a]-flow[a]
    value=3.-np.linalg.norm(v,axis=1)/5.
    p,q=parent[a],child[b];has_p=p>=0;has_q=q>=0
    vp=xyz[a]-xyz[p.clip(0)]-flow[p.clip(0)]
    vq=xyz[q.clip(0)]-xyz[b]-flow[b]
    value-=CONFIG['adjacent_weight']*(has_p*penalty(vp,v)+has_q*penalty(v,vq))
    pp=parent[p.clip(0)];qq=child[q.clip(0)]
    vpp=xyz[p.clip(0)]-xyz[pp.clip(0)]-flow[pp.clip(0)]
    vqq=xyz[qq.clip(0)]-xyz[q.clip(0)]-flow[q.clip(0)]
    value-=CONFIG['long_weight']*((has_p&(pp>=0))*penalty(vpp,v)+(has_p&has_q)*penalty(vp,vq)+(has_q&(qq>=0))*penalty(v,vqq))
    # Explicit tie-break identical to maximum_vote_edges, accounted in objective.
    return value

def objective(coords,flow,edges):
    if not len(edges):return 0.
    xyz=coords[:,1:]*SCALE;_,child=links(len(coords),edges);a,b=edges.T
    v=xyz[b]-xyz[a]-flow[a];c=child[b];d=child[c.clip(0)]
    value=(3.-np.linalg.norm(v,axis=1)/5.+.01/(1+np.linalg.norm(xyz[b]-xyz[a],axis=1))).sum()
    w=xyz[c.clip(0)]-xyz[b]-flow[b]
    u=xyz[d.clip(0)]-xyz[c.clip(0)]-flow[c.clip(0)]
    value-=CONFIG['adjacent_weight']*np.sum((c>=0)*penalty(v,w))
    value-=CONFIG['long_weight']*np.sum(((c>=0)&(d>=0))*penalty(v,u))
    return float(value)

def optimize(coords,flow,edges,initial):
    selected=initial.copy();history=[objective(coords,flow,selected)]
    times=np.unique(coords[edges[:,0],0]) if len(edges) else []
    blocks={t:edges[coords[edges[:,0],0]==t] for t in times}
    for sweep in range(CONFIG['sweeps']):
        changes=0
        for t in (times if sweep%2==0 else times[::-1]):
            candidates=blocks[t];utility=edge_utilities(coords,flow,candidates,selected)
            good=utility>0;chosen=maximum_vote_edges(coords,candidates[good],utility[good])
            old=selected[coords[selected[:,0],0]==t]
            if set(map(tuple,chosen))==set(map(tuple,old)):continue
            rest=selected[coords[selected[:,0],0]!=t]
            proposal=np.concatenate((rest,chosen));score=objective(coords,flow,proposal)
            # Enforce improvement of the declared full-sequence energy.
            if score+1e-7>=history[-1]:
                selected=proposal;history.append(score);changes+=1
        if not changes:break
    selected=selected[np.lexsort((selected[:,1],selected[:,0]))] if len(selected) else selected
    return selected,dict(initial_objective=history[0],final_objective=history[-1],accepted_blocks=len(history)-1,
        objective_nondecreasing=bool(np.all(np.diff(history)>=-1e-7)))

def track(coords,initial):
    flow,flow_report=tissue_flow(coords);edges=candidate_edges(coords,flow)
    empty=np.empty((0,2),np.int64);utility=edge_utilities(coords,flow,edges,empty);good=utility>0
    flow_initial=maximum_vote_edges(coords,edges[good],utility[good])
    arms=[];reports={}
    # Include original edges so the old assignment remains a feasible starting point.
    edges=np.unique(np.concatenate((edges,initial)),axis=0)
    for name,start in [('geometric',initial),('registered',flow_initial)]:
        output,report=optimize(coords,flow,edges,start);arms.append(output);reports[name]=report
    index=int(reports['registered']['final_objective']>reports['geometric']['final_objective'])
    return arms[index],dict(config=CONFIG,flow=flow_report,starts=reports,selected_start=['geometric','registered'][index],
        candidate_edges=len(edges),annotations_used=False,scope='Two-start coordinate descent; not global optimum or full MHT reproduction')
