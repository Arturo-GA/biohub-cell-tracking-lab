"""Factored trajectory hypotheses from detections alone; no reference graph or GT.

Division pairs are bit-packed, so all eligible pairs are represented without
enumerating the exponential number of future paths. A two-unit residual flow
materializes vertex-disjoint daughter trajectories for any eligible pair.
"""
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .detector_proposals import SCALE, nms


@dataclass(frozen=True)
class DAGConfig:
    merge_um: float = 1.2
    mother_radius_um: float = 20.
    max_daughters: int = 16
    sister_min_um: float = 1.5
    sister_max_um: float = 20.
    midpoint_max_um: float = 10.
    continuation_um: float = 10.
    max_continuations: int = 8
    horizon: int = 4
    sample_queries: int = 128


def select_development(names, excluded, per_group=24):
    """Filename-only selection. The remaining videos are not inspected here."""
    if len(names)!=len(set(names)):raise ValueError('Duplicate dataset names')
    groups={}
    for name in sorted(set(names)-set(excluded)):
        groups.setdefault(name.split('_')[0],[]).append(name)
    if len(groups)!=2 or any(len(v)<per_group for v in groups.values()):
        raise ValueError('Insufficient videos for the two acquisition groups')
    selected=[]
    for group,values in sorted(groups.items()):
        ordered=sorted(values,key=lambda name:(hashlib.sha256(('biohub-detection-dag-dev-v1:'+name).encode()).hexdigest(),name))
        selected.extend(ordered[:per_group])
    return sorted(selected)


def merge_sources(sources,shape,config=DAGConfig()):
    points=[];values=[];origins=[]
    for index,(coords,scores) in enumerate(sources,1):
        coords=np.asarray(coords,np.float32);scores=np.asarray(scores,np.float32)
        if coords.shape!=(len(scores),4) or not np.isfinite(coords).all() or not np.isfinite(scores).all():
            raise ValueError('Invalid raw detector proposals')
        if np.any(coords<0) or np.any(coords>np.asarray(shape)-1) or np.any(coords[:,0]!=np.rint(coords[:,0])):
            raise ValueError('Out-of-bounds proposals or fractional time')
        points.append(np.rint(coords));values.append(scores);origins.append(np.full(len(coords),index,np.int8))
    if not points:raise ValueError('At least one detector source is required')
    coords=np.concatenate(points);scores=np.concatenate(values);origin=np.concatenate(origins);keep=[]
    for t in range(shape[0]):
        ids=np.flatnonzero(coords[:,0]==t)
        keep.extend(ids[nms(coords[ids,1:],scores[ids],config.merge_um)])
    keep=np.asarray(keep,np.int64)
    return coords[keep].astype(np.int32),scores[keep],origin[keep]


def neighbor_table(coords,k,radius):
    """Directed adjacent-frame edges with a fixed physical radius and degree cap."""
    output=np.full((len(coords),k),-1,np.int32)
    for frame in np.unique(coords[:,0]):
        source=np.flatnonzero(coords[:,0]==frame);target=np.flatnonzero(coords[:,0]==frame+1)
        if not len(target):continue
        distances,local=cKDTree(coords[target,1:]*SCALE).query(coords[source,1:]*SCALE,
            k=k,distance_upper_bound=radius)
        distances=distances.reshape(len(source),k);local=local.reshape(len(source),k)
        valid=np.isfinite(distances)
        rows,columns=np.nonzero(valid);output[source[rows],columns]=target[local[rows,columns]]
    return output


def pair_mask(coords,daughters,config=DAGConfig()):
    slots=np.array(list(combinations(range(config.max_daughters),2)),np.int16)
    packed=np.zeros((len(coords),(len(slots)+7)//8),np.uint8);counts=np.zeros(len(coords),np.int16)
    for start in range(0,len(coords),1024):
        candidates=daughters[start:start+1024]
        a=candidates[:,slots[:,0]];b=candidates[:,slots[:,1]]
        pa=coords[a.clip(0),1:]*SCALE;pb=coords[b.clip(0),1:]*SCALE
        separation=np.linalg.norm(pa-pb,axis=-1)
        midpoint=np.linalg.norm((pa+pb)/2-coords[start:start+1024,None,1:]*SCALE,axis=-1)
        allowed=(a>=0)&(b>=0)&(separation>=config.sister_min_um)&(separation<=config.sister_max_um)&(midpoint<=config.midpoint_max_um)
        packed[start:start+1024]=np.packbits(allowed,axis=1,bitorder='little')
        counts[start:start+1024]=allowed.sum(1)
    return slots,packed,counts


def build_dag(coords,scores,origin,shape,config=DAGConfig()):
    coords=np.asarray(coords,np.int32)
    if coords.shape!=(len(scores),4) or len(origin)!=len(coords):raise ValueError('Malformed detection pool')
    if np.any(coords<0) or np.any(coords>np.asarray(shape)-1):raise ValueError('Invalid coordinates')
    if len({tuple(v) for v in coords})!=len(coords):raise ValueError('Duplicate detections')
    daughters=neighbor_table(coords,config.max_daughters,config.mother_radius_um)
    continuation=neighbor_table(coords,config.max_continuations,config.continuation_um)
    slots,packed,counts=pair_mask(coords,daughters,config)
    return dict(coords=coords,scores=np.asarray(scores,np.float32),origin=np.asarray(origin,np.int8),
        shape=np.asarray(shape,np.int32),daughters=daughters,continuation=continuation,
        pair_slots=slots,pair_bits=packed,pair_counts=counts)


def pair_allowed(graph,mother,a,b):
    indices=graph['daughters'][mother]
    ia=np.flatnonzero(indices==a);ib=np.flatnonzero(indices==b)
    if not len(ia) or not len(ib) or a==b:return False
    left,right=sorted((int(ia[0]),int(ib[0])))
    slot=np.flatnonzero((graph['pair_slots'][:,0]==left)&(graph['pair_slots'][:,1]==right))
    if len(slot)!=1:return False
    i=int(slot[0]);return bool((graph['pair_bits'][mother,i//8]>>(i%8))&1)


def two_paths(graph,a,b,target_frame):
    """Exact two-path feasibility by residual flow, including rerouting conflicts.

    Paths have one observed detection per frame; no invented coordinates or
    minimum history. The requested endpoint is a time, not an existing track.
    """
    coords=graph['coords'];following=graph['continuation']
    if a==b or min(a,b)<0 or max(a,b)>=len(coords):return None
    start=int(coords[a,0])
    if coords[b,0]!=start or target_frame<start or target_frame>=graph['shape'][0]:return None
    reachable={int(a),int(b)};frontier=set(reachable)
    for frame in range(start,target_frame):
        next_nodes={int(v) for node in frontier for v in following[node] if v>=0}
        if len(next_nodes)<2:return None
        if any(coords[n,0]!=frame+1 for n in next_nodes):raise ValueError('Non-consecutive DAG edge')
        reachable.update(next_nodes);frontier=next_nodes
    # Split every detection into an in/out vertex with capacity one.
    residual={};original=[]
    def add(u,v):
        residual.setdefault(u,{})[v]=1;residual.setdefault(v,{})[u]=0;original.append((u,v))
    source=-2;sink=-1
    for node in sorted(reachable):
        add(2*node,2*node+1)
        if coords[node,0]==target_frame:add(2*node+1,sink)
        else:
            for target in following[node]:
                if int(target) in reachable:add(2*node+1,2*int(target))
    add(source,2*int(a));add(source,2*int(b))
    for _ in range(2):
        previous={source:None};queue=deque([source])
        while queue and sink not in previous:
            u=queue.popleft()
            for v,capacity in residual[u].items():
                if capacity and v not in previous:previous[v]=u;queue.append(v)
        if sink not in previous:return None
        v=sink
        while previous[v] is not None:
            u=previous[v];residual[u][v]-=1;residual[v][u]+=1;v=u
    used={u:v for u,v in original if u>=0 and u%2==1 and residual[u][v]==0}
    paths=[]
    for initial in (int(a),int(b)):
        path=[initial]
        while coords[path[-1],0]<target_frame:
            vertex=used[2*path[-1]+1]
            if vertex<0:raise ValueError('Premature trajectory end')
            path.append(vertex//2)
        paths.append(path)
    if set(paths[0])&set(paths[1]):raise ValueError('Flow paths share a detection')
    return paths


def sample_trajectories(graph,name,config=DAGConfig()):
    """Exercise the generator on label-free random pairs, before any evaluation."""
    seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);counts=graph['pair_counts'].astype(np.int64)
    total=int(counts.sum());queries=[];paths=[]
    if not total:return np.empty((0,3),np.int32),np.empty((0,2,config.horizon),np.int32)
    cumulative=np.cumsum(counts)
    for draw in rng.integers(total,size=config.sample_queries):
        mother=int(np.searchsorted(cumulative,draw,side='right'))
        allowed=np.flatnonzero(np.unpackbits(graph['pair_bits'][mother],bitorder='little')[:len(graph['pair_slots'])])
        slot=int(allowed[int(draw-(cumulative[mother]-counts[mother]))])
        a,b=graph['daughters'][mother,graph['pair_slots'][slot]]
        end=min(int(graph['coords'][mother,0])+config.horizon,int(graph['shape'][0])-1)
        result=two_paths(graph,int(a),int(b),end);queries.append([mother,int(a),int(b)])
        stored=np.full((2,config.horizon),-1,np.int32)
        if result is not None:
            for i,path in enumerate(result):stored[i,:len(path)]=path
        paths.append(stored)
    return np.asarray(queries,np.int32),np.asarray(paths,np.int32)


def save_dag(graph,output,name,config=DAGConfig()):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'graph.npz';np.savez_compressed(path,**graph)
    triples,paths=sample_trajectories(graph,name,config)
    np.savez_compressed(output/'trajectory_samples.npz',triples=triples,paths=paths)
    receipt=dict(config=asdict(config),nodes=len(graph['coords']),
        mother_daughter_edges=int((graph['daughters']>=0).sum()),
        continuation_edges=int((graph['continuation']>=0).sum()),
        division_pairs=int(graph['pair_counts'].sum()),mothers_with_pairs=int((graph['pair_counts']>0).sum()),
        graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sample_queries=len(triples),sample_queries_with_paths=int((paths[:,0,0]>=0).sum()) if len(paths) else 0,
        annotations_read=False,reference_graph_used=False,
        representation='Packed division pairs and adjacent-frame detection DAG; all eligible pairs are queryable, sampled paths are not a final selection.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
