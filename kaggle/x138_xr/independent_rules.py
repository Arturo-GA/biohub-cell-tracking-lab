"""E069: label-free postprocessing of the final c3 graph.

All coordinates are native z,y,x voxels; distances below are micrometres.
The linker only proposes degree-preserving swaps, never new divisions.
"""
import collections
import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625])


def graph_context(nodes, edges):
    pos = {k: np.array([v[a] for a in ('z', 'y', 'x')], float)*SCALE for k,v in nodes.items()}
    pred, succ = collections.defaultdict(list), collections.defaultdict(list)
    for e in edges:
        s,d = int(e['source_id']),int(e['target_id'])
        assert s in nodes and d in nodes and int(nodes[d]['t']) > int(nodes[s]['t'])
        succ[s].append(d); pred[d].append(s)
    return pos,pred,succ


def swap_candidates(nodes, edges, radius=12., max_step=10.):
    """Identical broad candidate set for geometric and neural controls."""
    pos,pred,succ = graph_context(nodes,edges)
    by_frame=collections.defaultdict(list); contexts={}
    for a in sorted(nodes):
        if len(pred[a])!=1 or len(succ[a])!=1: continue
        p,d=pred[a][0],succ[a][0]
        if len(succ[p])!=1 or len(pred[d])!=1 or len(succ[d])!=1: continue
        q=succ[d][0]
        if len(pred[q])!=1: continue
        if [int(nodes[k]['t']) for k in (p,a,d,q)] != list(range(int(nodes[a]['t'])-1,int(nodes[a]['t'])+3)): continue
        contexts[a]=(p,d,q); by_frame[int(nodes[a]['t'])].append(a)
    candidates=[]
    for t,ids in sorted(by_frame.items()):
        if len(ids)<2: continue
        tree=cKDTree([pos[k] for k in ids])
        for i,j in sorted(tree.query_pairs(radius)):
            a,b=ids[i],ids[j]; pa,da,qa=contexts[a]; pb,db,qb=contexts[b]
            context=(pa,a,da,qa,pb,b,db,qb)
            if len(set(context))!=8: continue
            if max(np.linalg.norm(pos[db]-pos[a]),np.linalg.norm(pos[da]-pos[b]))>max_step: continue
            def errors(p,s,d,q):
                return np.array([np.linalg.norm(pos[d]-2*pos[s]+pos[p]),np.linalg.norm(pos[s]-2*pos[d]+pos[q])])
            olda,oldb=errors(pa,a,da,qa),errors(pb,b,db,qb)
            newa,newb=errors(pa,a,db,qb),errors(pb,b,da,qa)
            old,new=float(sum(olda+oldb)),float(sum(newa+newb))
            candidates.append(dict(t=t,a=a,b=b,da=da,db=db,context=list(context),
                geom_old=old,geom_new=new,geom_gain=old-new,
                geom_both=bool(np.all(newa+newb<olda+oldb) and sum(newa)<=sum(olda) and sum(newb)<=sum(oldb))))
    return candidates


def apply_swaps(nodes, edges, candidates, mode='neural_strict', cap_fraction=.002):
    pos,_,_=graph_context(nodes,edges)
    eligible=[]
    for c in candidates:
        if mode=='geometry':
            good=c['geom_both'] and c['geom_gain']>=1. and c['geom_new']<=.75*c['geom_old']
            priority=c['geom_gain']
        else:
            if 'new_min_logit' not in c: continue
            strict=mode=='neural_strict'
            good=(c['new_min_logit']>0 and c['mutual_new'] and
                  c['min_row_gain']>=(2. if strict else 1.) and
                  c['min_col_gain']>=(2. if strict else 1.) and
                  c['geom_new']<=c['geom_old']+(0. if strict else 1.))
            priority=c['min_row_gain']+c['min_col_gain']
        if good: eligible.append((-priority,c['t'],c['a'],c['b'],c))
    used=set(); drop=set(); added=[]; accepted=[]
    limit=max(1,int(len(edges)*cap_fraction))
    for *_,c in sorted(eligible,key=lambda x:x[:4]):
        if len(accepted)>=limit: break
        if used.intersection(c['context']): continue
        a,b,da,db=(c[k] for k in ('a','b','da','db'))
        drop.update(((a,da),(b,db))); used.update(c['context']);accepted.append(c)
        for s,d in ((a,db),(b,da)):
            added.append(dict(source_id=s,target_id=d,edge_prob=None,
                              distance_um=float(np.linalg.norm(pos[d]-pos[s])),independent_swap=1))
    result=[dict(e) for e in edges if (int(e['source_id']),int(e['target_id'])) not in drop]+added
    for key in ('source_id','target_id'):
        assert collections.Counter(e[key] for e in result)==collections.Counter(e[key] for e in edges)
    return nodes,result,dict(candidates=len(candidates),eligible=len(eligible),swaps=len(accepted),accepted=accepted)


def persistent_duplicates(nodes, edges, radius=2., min_overlap=8, max_length=24,
                          max_confidence=.65, confidence_gap=.1, coverage=.9):
    """Remove only a weaker complete linear component covered by a stronger one.

    Both components must be division-free. Never remove a partial branch or
    splice tracks. Proximity must persist, cover the weaker track, and be
    accompanied by a confident alternative and similar motion.
    """
    pos,pred,succ=graph_context(nodes,edges)
    components=[];owner={};seen=set()
    for root in sorted(nodes):
        if root in seen: continue
        stack=[root];comp=[]
        while stack:
            k=stack.pop()
            if k in seen: continue
            seen.add(k);comp.append(k);stack.extend(pred[k]+succ[k])
        if len(comp)<min_overlap or any(len(pred[k])>1 or len(succ[k])>1 for k in comp): continue
        ts={int(nodes[k]['t']):k for k in comp}
        if len(ts)!=len(comp) or max(ts)-min(ts)+1!=len(ts): continue
        idx=len(components);components.append(dict(ids=comp,times=ts,confidence=[]))
        for k in comp: owner[k]=idx
    for e in edges:
        s=int(e['source_id']);p=e.get('edge_prob')
        if s in owner and p is not None and np.isfinite(p): components[owner[s]]['confidence'].append(float(p))
    for c in components:
        c['confidence']=float(np.mean(c['confidence'])) if c['confidence'] else None
    frames=collections.defaultdict(list)
    for k in owner: frames[int(nodes[k]['t'])].append(k)
    near=collections.Counter()
    for ids in frames.values():
        if len(ids)<2: continue
        for i,j in cKDTree([pos[k] for k in ids]).query_pairs(radius):
            a,b=owner[ids[i]],owner[ids[j]]
            if a!=b: near[tuple(sorted((a,b)))]+=1
    candidates=[]
    for (a,b),count in sorted(near.items()):
        if count<min_overlap: continue
        ca,cb=components[a],components[b]
        if ca['confidence'] is None or cb['confidence'] is None: continue
        weak,strong=(a,b) if (ca['confidence'],len(ca['ids']),a)<(cb['confidence'],len(cb['ids']),b) else (b,a)
        w,s=components[weak],components[strong]
        if len(w['ids'])>max_length or len(s['ids'])<len(w['ids']): continue
        if w['confidence']>max_confidence or s['confidence']-w['confidence']<confidence_gap: continue
        ts=sorted(set(w['times'])&set(s['times']))
        if len(ts)<min_overlap or len(ts)/len(w['ids'])<coverage: continue
        offsets=np.array([pos[w['times'][t]]-pos[s['times'][t]] for t in ts])
        dist=np.linalg.norm(offsets,axis=1)
        if np.quantile(dist,.9)>radius or np.mean(dist<=radius)<coverage: continue
        velocity=np.linalg.norm(np.diff(offsets,axis=0),axis=1)
        if np.quantile(velocity,.9)>.75: continue
        candidates.append(dict(weak=weak,strong=strong,weak_nodes=len(w['ids']),strong_nodes=len(s['ids']),
                               weak_confidence=w['confidence'],strong_confidence=s['confidence'],
                               overlap=len(ts),distance90=float(np.quantile(dist,.9))))
    removed_components=set();protected=set();removed=set();accepted=[]
    for c in sorted(candidates,key=lambda r:(r['weak_confidence'],-r['strong_confidence'],r['weak'])):
        if c['weak'] in removed_components or c['weak'] in protected or c['strong'] in removed_components: continue
        removed_components.add(c['weak']);protected.add(c['strong'])
        removed.update(components[c['weak']]['ids']);accepted.append(c)
    return ({k:v for k,v in nodes.items() if k not in removed},
            [dict(e) for e in edges if int(e['source_id']) not in removed and int(e['target_id']) not in removed],
            dict(linear_components=len(components),near_component_pairs=len(near),candidates=len(candidates),
                 removed_tracks=len(accepted),removed_nodes=len(removed),accepted=accepted))


def join_candidates(nodes,edges,max_distance=5.,motion_error=1.5):
    """Consecutive fragmented tracks with two-frame motion on both sides."""
    pos,pred,succ=graph_context(nodes,edges)
    starts=collections.defaultdict(list);ends=collections.defaultdict(list);context={}
    for k in sorted(nodes):
        t=int(nodes[k]['t'])
        if not pred[k] and len(succ[k])==1:
            q=succ[k][0]
            if len(pred[q])==1 and len(succ[q])==1:
                qq=succ[q][0]
                if len(pred[qq])==1 and [int(nodes[x]['t']) for x in (k,q,qq)]==[t,t+1,t+2]:
                    starts[t].append(k);context[k]=(q,qq)
        if not succ[k] and len(pred[k])==1:
            p=pred[k][0]
            if len(succ[p])==1 and len(pred[p])==1:
                pp=pred[p][0]
                if len(succ[pp])==1 and [int(nodes[x]['t']) for x in (pp,p,k)]==[t-2,t-1,t]:
                    ends[t].append(k);context[k]=(p,pp)
    candidates=[]
    for t,ss in sorted(ends.items()):
        ds=starts.get(t+1,[])
        if not ds:continue
        tree=cKDTree([pos[d] for d in ds])
        for s in ss:
            p,pp=context[s];vf=(pos[s]-pos[pp])/2
            for j in tree.query_ball_point(pos[s],max_distance):
                d=ds[j];q,qq=context[d];vb=(pos[qq]-pos[d])/2
                ef=float(np.linalg.norm(pos[d]-pos[s]-vf));eb=float(np.linalg.norm(pos[d]-pos[s]-vb))
                if max(ef,eb)>motion_error:continue
                candidates.append(dict(t=t,s=s,d=d,context=[pp,p,s,d,q,qq],motion_error=max(ef,eb)))
    return candidates


def apply_joins(nodes,edges,candidates,neural=True):
    current={(c['s'],c['d']):c for c in join_candidates(nodes,edges)}
    eligible=[]
    for c in candidates:
        if (c['s'],c['d']) not in current or current[(c['s'],c['d'])]['context']!=c['context']:continue
        if neural and not (c.get('mutual',False) and c.get('row_margin',-1)>=2 and c.get('col_margin',-1)>=2):continue
        eligible.append(c)
    used=set();added=[];accepted=[]
    for c in sorted(eligible,key=lambda c:(-(min(c.get('row_margin',0),c.get('col_margin',0))) if neural else c['motion_error'],c['s'],c['d'])):
        if used.intersection(c['context']):continue
        added.append(dict(source_id=c['s'],target_id=c['d'],edge_prob=None,independent_join=1))
        used.update(c['context']);accepted.append(c)
    return nodes,[dict(e) for e in edges]+added,dict(candidates=len(candidates),eligible=len(eligible),joins=len(added),accepted=accepted)


def fork_veto_candidates(nodes,edges):
    _,pred,succ=graph_context(nodes,edges)
    lookup={(int(e['source_id']),int(e['target_id'])):e for e in edges}
    result=[]
    for s,children in sorted(succ.items()):
        if len(children)!=2:continue
        uncertain=[d for d in children if lookup[s,d].get('edge_prob') is None or not np.isfinite(lookup[s,d]['edge_prob'])]
        if len(uncertain)!=1:continue
        d=uncertain[0];keep=next(k for k in children if k!=d);t=int(nodes[s]['t'])
        if any(len(pred[k])!=1 or int(nodes[k]['t'])!=t+1 for k in children):continue
        result.append(dict(t=t,s=s,d=d,keep=keep))
    return result


def apply_fork_veto(nodes,edges,candidates):
    allowed={(c['s'],c['d'],c['keep']) for c in fork_veto_candidates(nodes,edges)}
    accepted=[c for c in candidates if (c['s'],c['d'],c['keep']) in allowed
              and c.get('weak_logit',0)<-2 and c.get('keep_logit',0)>2 and c.get('keep_top',False)]
    drop={(c['s'],c['d']) for c in accepted}
    return nodes,[dict(e) for e in edges if (int(e['source_id']),int(e['target_id'])) not in drop],dict(candidates=len(candidates),fork_vetoes=len(drop),accepted=accepted)
