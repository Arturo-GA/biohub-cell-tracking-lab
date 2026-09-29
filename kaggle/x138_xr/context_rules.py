"""Geometry-only hypotheses on c3: temporal swaps, context-aware cuts, smoothing.

No labels, no new detections, and no change to safe-division thresholds.
"""
import collections as _ctx_col
import numpy as np
from scipy.spatial import cKDTree


def ctx_graph(nodes, edges):
    pos = {k: np.array([v['z'], v['y'], v['x']], dtype=float) * np.array([1.625, .40625, .40625]) for k,v in nodes.items()}
    pred, succ = _ctx_col.defaultdict(list), _ctx_col.defaultdict(list)
    for e in edges:
        s,d=int(e['source_id']),int(e['target_id'])
        if s in nodes and d in nodes and int(nodes[d]['t'])==int(nodes[s]['t'])+1:
            succ[s].append(d);pred[d].append(s)
    return pos,pred,succ


def temporal_swaps(nodes, edges, relative_gain=.25, absolute_gain=1.0, radius=10.0):
    """Swap two uncertain links if forward AND backward motion improve.

All node positions, edge count and node degrees stay fixed. Protect forks and
use disjoint four-frame contexts to prevent a cascade of incompatible swaps.
"""
    pos,pred,succ=ctx_graph(nodes,edges)
    frames=_ctx_col.defaultdict(list)
    contexts={}
    for s in sorted(nodes):
        if len(pred[s])!=1 or len(succ[s])!=1:continue
        p=pred[s][0];d=succ[s][0]
        if len(succ[p])!=1 or len(pred[d])!=1 or len(succ[d])!=1:continue
        q=succ[d][0]
        if len(pred[q])!=1:continue
        contexts[s]=(p,d,q)
        frames[int(nodes[s]['t'])].append(s)
    candidates=[]
    for t,ids in frames.items():
        if len(ids)<2:continue
        tree=cKDTree(np.array([pos[s] for s in ids]))
        for i,j in sorted(tree.query_pairs(radius)):
            a,b=ids[i],ids[j];pa,da,qa=contexts[a];pb,db,qb=contexts[b]
            if len({pa,a,da,qa,pb,b,db,qb})!=8:continue
            def errors(s,p,d,q):
                return np.linalg.norm(pos[d]-(2*pos[s]-pos[p])),np.linalg.norm(pos[s]-(2*pos[d]-pos[q]))
            old_a=errors(a,pa,da,qa);old_b=errors(b,pb,db,qb)
            new_a=errors(a,pa,db,qb);new_b=errors(b,pb,da,qa)
            if max(np.linalg.norm(pos[db]-pos[a]),np.linalg.norm(pos[da]-pos[b]))>10:continue
            old_f,old_bk=np.add(old_a,old_b);new_f,new_bk=np.add(new_a,new_b)
            old,new=float(old_f+old_bk),float(new_f+new_bk)
            if new_f>=old_f or new_bk>=old_bk:continue
            if old-new<absolute_gain or new>old*(1-relative_gain):continue
            # Avoid paying for an improvement in only one of the two tracks.
            if sum(new_a)>sum(old_a) or sum(new_b)>sum(old_b):continue
            candidates.append((-(old-new),a,b,da,db,{pa,a,da,qa,pb,b,db,qb}))
    used=set();drop=set();added=[]
    limit=max(1,int(len(edges)*.002))
    for _,a,b,da,db,context in sorted(candidates,key=lambda x:x[:5]):
        if len(added)//2>=limit:break
        if used & context:continue
        for s,d in ((a,db),(b,da)):
            added.append(dict(source_id=s,target_id=d,edge_prob=None,distance_um=float(np.linalg.norm(pos[s]-pos[d])),context_swap=1))
        drop.update(((a,da),(b,db)));used.update(context)
    return nodes,[dict(e) for e in edges if (int(e['source_id']),int(e['target_id'])) not in drop]+added,dict(swap_pairs=len(added)//2,swap_candidates=len(candidates))


def context_cuts(nodes, edges, protect='both', residual=1.5):
    """Same c3 cuts except well-supported motions may be retained.

Neighbour flow is a veto when >=3 reliable local links are available. It is
never sufficient on its own; an individual trajectory must also be consistent.
"""
    pos,pred,succ=ctx_graph(nodes,edges)
    reliable=_ctx_col.defaultdict(list)
    for e in edges:
        s,d=int(e['source_id']),int(e['target_id'])
        if s not in pos or d not in pos:continue
        if len(succ[s])==1 and len(pred[d])==1 and np.linalg.norm(pos[d]-pos[s])<8:
            reliable[int(nodes[s]['t'])].append((s,pos[d]-pos[s]))
    trees={t:(cKDTree(np.array([pos[s] for s,v in es])),es) for t,es in reliable.items() if len(es)>=4}
    def supported(s,d,is_end):
        if len(succ[s])!=1 or len(pred[s])!=1 or len(pred[d])!=1:return False
        p=pred[s][0]
        if len(succ[p])!=1 or len(pred[p])!=1:return False
        pp=pred[p][0]
        if len(succ[pp])!=1:return False
        v=.5*((pos[s]-pos[p])+(pos[p]-pos[pp]))
        if np.linalg.norm((pos[d]-pos[s])-v)>residual:return False
        if not is_end:
            if len(succ[d])!=1:return False
            q=succ[d][0]
            if len(pred[q])!=1 or np.linalg.norm((pos[q]-pos[d])-(pos[d]-pos[s]))>residual:return False
        t=int(nodes[s]['t'])
        if t in trees:
            tree,rs=trees[t];idx=tree.query_ball_point(pos[s],25)
            vs=[rs[i][1] for i in idx if rs[i][0]!=s and np.linalg.norm(pos[rs[i][0]]-pos[s])>1.5]
            if len(vs)>=3 and np.linalg.norm((pos[d]-pos[s])-np.median(vs,axis=0))>2.5:return False
        return True
    kept=[];stats=dict(long_cut=0,long_saved=0,end_cut=0,end_saved=0)
    for e in edges:
        s,d=int(e['source_id']),int(e['target_id']);p=e.get('edge_prob')
        finite=p is not None and np.isfinite(float(p))
        long=not finite and float(e.get('distance_um',0) or 0)>8
        if long:
            if protect in ('both','long') and supported(s,d,False):stats['long_saved']+=1
            else:stats['long_cut']+=1;continue
        kept.append(dict(e))
    # c3 computes endpoint status after removing the long edges.
    _,_,succ2=ctx_graph(nodes,kept)
    result=[]
    for e in kept:
        s,d=int(e['source_id']),int(e['target_id']);p=e.get('edge_prob')
        end=p is not None and np.isfinite(float(p)) and float(p)<.5 and not succ2[d]
        if end:
            if protect in ('both','end') and supported(s,d,True):stats['end_saved']+=1
            else:stats['end_cut']+=1;continue
        result.append(e)
    return nodes,result,stats


def topology_smoothing(nodes, edges, smoothed, weight=.3):
    """Reduce reference linefit strength at forks and their immediate daughters."""
    _,_,succ=ctx_graph(nodes,edges)
    affected={s for s in succ if len(succ[s])==2}
    affected |= {d for s in list(affected) for d in succ[s]}
    out={k:dict(v) for k,v in smoothed.items()}
    ratio=float(weight)/.8  # upstream c3 linefit weight is fixed at .8
    for k in affected:
        for axis in ('z','y','x'):
            out[k][axis]=float(nodes[k][axis])+ratio*(float(smoothed[k][axis])-float(nodes[k][axis]))
    return out,len(affected)
