"""CPU hypothesis: extend short daughter tracks using existing detections.

Inspired by complete_division_lineages in amanatar/optimized-biohub-max-score
(reviewed 2026-09-26). Independent implementation; no labels or image inference.
"""
import collections
import numpy as np


def extend_daughters(nodes, edges, radius_um=4.0, min_length=8, max_added_fraction=0.002):
    if not nodes or not edges:
        return nodes, list(edges)
    pos = {k: np.array([v['z'],v['y'],v['x']],dtype=float)*np.array([1.625,0.40625,0.40625]) for k,v in nodes.items()}
    by_t = collections.defaultdict(list)
    for k in sorted(nodes):
        by_t[int(nodes[k]['t'])].append(k)
    succ, pred = collections.defaultdict(list), collections.defaultdict(list)
    for e in edges:
        s,d=int(e['source_id']),int(e['target_id'])
        succ[s].append(d);pred[d].append(s)
    result=list(edges);cap=max(1,int(len(edges)*max_added_fraction));added=0
    forks=sorted(k for k in nodes if len(succ[k])==2)
    for fork in forks:
        for daughter in list(succ[fork]):
            current=daughter;length=1;seen={current}
            while len(succ[current])==1:
                nxt=succ[current][0]
                if nxt in seen or len(pred[nxt])!=1 or int(nodes[nxt]['t'])!=int(nodes[current]['t'])+1:
                    break
                current=nxt;seen.add(nxt);length+=1
            while length<min_length and not succ[current] and added<cap:
                t=int(nodes[current]['t'])
                options=[(float(np.linalg.norm(pos[k]-pos[current])),k) for k in by_t[t+1] if not pred[k]]
                if not options:break
                dist,target=min(options)
                if dist>radius_um:break
                # Mutual nearest among available ends; do not steal a continuation.
                reverse=min((float(np.linalg.norm(pos[k]-pos[target])),k) for k in by_t[t] if not succ[k])
                if reverse[1]!=current:break
                result.append(dict(source_id=current,target_id=target,edge_prob=None,distance_um=dist,lineage_repair=1))
                succ[current].append(target);pred[target].append(current)
                current=target;length+=1;added+=1
    return nodes,result
