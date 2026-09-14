"""Post-freeze coverage ceilings. Annotation witnesses are never predictions."""
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from .detector_proposals import SCALE
from .detection_dag import DAGConfig, pair_allowed, two_paths


@dataclass(frozen=True)
class CoverageGate:
    min_events: int = 20
    min_full_context_events: int = 20
    min_events_per_group: int = 5
    min_full_context_per_group: int = 5
    min_pair_coverage: float = .90
    min_path_coverage: float = .90
    min_annotated_path_coverage: float = .85
    match_um: float = 7.


def annotated_paths(graph,a,b,allowed_a,allowed_b):
    """Exact two-branch state search with GT neighborhoods, for auditing only.

    This checks whether paths already represented by the frozen graph can
    follow both annotated daughters. It does not modify or rank hypotheses.
    """
    if a==b or a not in allowed_a[0] or b not in allowed_b[0]:return None
    states={(int(a),int(b))};history=[]
    for index in range(1,len(allowed_a)):
        previous={}
        for left,right in sorted(states):
            aa=[int(n) for n in graph['continuation'][left] if n>=0 and n in allowed_a[index]]
            bb=[int(n) for n in graph['continuation'][right] if n>=0 and n in allowed_b[index]]
            for x in aa:
                for y in bb:
                    if x!=y:previous.setdefault((x,y),(left,right))
        if not previous:return None
        history.append(previous);states=set(previous)
    state=min(states);pairs=[state]
    for previous in reversed(history):state=previous[state];pairs.append(state)
    return np.asarray(pairs[::-1],np.int32).T.tolist()


def evaluate_graph(graph,truth,edges,config=DAGConfig(),match_um=7.):
    coords=graph['coords'];truth=np.asarray(truth);children=defaultdict(list)
    for source,target in edges:
        if truth[target,0]!=truth[source,0]+1:raise ValueError('Non-consecutive annotation edge')
        children[int(source)].append(int(target))
    if any(len(v)>2 for v in children.values()):raise ValueError('Nonbinary annotation')
    frames={int(t):np.flatnonzero(coords[:,0]==t) for t in np.unique(coords[:,0])}
    trees={t:cKDTree(coords[ids,1:]*SCALE) for t,ids in frames.items()}
    def near(point):
        t=int(point[0])
        if t not in trees:return set()
        return set(map(int,frames[t][trees[t].query_ball_point(point[1:]*SCALE,match_um)]))
    counts=dict(events=0,centers_covered=0,pairs_covered=0,paths_covered=0,
        full_context_events=0,annotated_paths_covered=0)
    records=[];flow_cache={}
    for mother,kids in sorted(children.items()):
        if len(kids)!=2:continue
        counts['events']+=1;t=int(truth[mother,0]);end=min(t+config.horizon,int(graph['shape'][0])-1)
        branches=[[kids[0]],[kids[1]]]
        for path in branches:
            while len(path)<config.horizon and len(children[path[-1]])==1:path.append(children[path[-1]][0])
        full=end==t+config.horizon and all(len(p)==config.horizon for p in branches)
        counts['full_context_events']+=int(full)
        mset=near(truth[mother]);aset=near(truth[kids[0]]);bset=near(truth[kids[1]])
        centers=bool(mset) and any(a!=b for a in aset for b in bset)
        counts['centers_covered']+=int(centers)
        triples=[]
        for m in sorted(mset):
            proposed=set(map(int,graph['daughters'][m]));aa=sorted(aset&proposed);bb=sorted(bset&proposed)
            triples.extend((m,a,b) for a in aa for b in bb if pair_allowed(graph,m,a,b))
        counts['pairs_covered']+=int(bool(triples))
        allowed_a=[near(truth[node]) for node in branches[0]] if full else []
        allowed_b=[near(truth[node]) for node in branches[1]] if full else []
        witness=None;annotated_witness=None
        for m,a,b in triples:
            if witness is None:
                key=(min(a,b),max(a,b),end)
                if key not in flow_cache:flow_cache[key]=two_paths(graph,key[0],key[1],end)
                if flow_cache[key] is not None:
                    witness=dict(triple=[m,a,b],paths=flow_cache[key] if a<b else flow_cache[key][::-1])
            if full and annotated_witness is None:
                paths=annotated_paths(graph,a,b,allowed_a,allowed_b)
                if paths is not None:annotated_witness=dict(triple=[m,a,b],paths=paths)
            if witness is not None and (not full or annotated_witness is not None):break
        counts['paths_covered']+=int(witness is not None)
        counts['annotated_paths_covered']+=int(annotated_witness is not None)
        reason=('missing_detection' if not centers else 'initial_pair_pruned' if not triples
                else 'no_disjoint_continuations' if witness is None else 'covered')
        records.append(dict(annotated_mother_index=mother,centers=centers,candidate_triples=len(triples),
            available_temporal_horizon=end-t,full_annotation_context=full,reason=reason,
            path_witness=witness,annotated_path_witness=annotated_witness))
    return counts,records


def readiness(per_video,gate=CoverageGate()):
    groups={};totals={}
    for name,row in per_video.items():
        group=groups.setdefault(name.split('_')[0],{})
        for key,value in row.items():
            group[key]=group.get(key,0)+value;totals[key]=totals.get(key,0)+value
    def rates(counts):
        return {key:counts[key]/counts[den] if counts[den] else None for key,den in (
            ('centers_covered','events'),('pairs_covered','events'),('paths_covered','events'),
            ('annotated_paths_covered','full_context_events'))}
    reasons=[]
    if totals.get('events',0)<gate.min_events:reasons.append('insufficient_division_events')
    if totals.get('full_context_events',0)<gate.min_full_context_events:reasons.append('insufficient_annotated_continuations')
    for name,counts in groups.items():
        if counts['events']<gate.min_events_per_group:reasons.append('insufficient_events_in_'+name)
        if counts['full_context_events']<gate.min_full_context_per_group:reasons.append('insufficient_context_in_'+name)
        group_rates=rates(counts)
        for key,threshold in [('pairs_covered',gate.min_pair_coverage),('paths_covered',gate.min_path_coverage),
                              ('annotated_paths_covered',gate.min_annotated_path_coverage)]:
            if group_rates[key] is None or group_rates[key]<threshold:reasons.append('below_'+key+'_in_'+name)
    rate=rates(totals) if totals else {}
    for key,threshold in [('pairs_covered',gate.min_pair_coverage),('paths_covered',gate.min_path_coverage),
                          ('annotated_paths_covered',gate.min_annotated_path_coverage)]:
        if rate.get(key) is None or rate[key]<threshold:reasons.append('below_'+key)
    return dict(gate=asdict(gate),totals=totals,rates=rate,groups={k:dict(counts=v,rates=rates(v)) for k,v in groups.items()},
        coverage_gate_passed=not reasons,reasons=reasons,
        decision='eligible_for_event_scoring_research' if not reasons else 'fix_coverage_or_collect_more_development_evidence_before_training',
        accuracy_demonstrated=False,training_started=False,leaderboard_submitted=False,
        scope='Coverage ceilings with annotation-conditioned witnesses, not prediction accuracy. Passing is necessary, not sufficient for training or submission.')
