"""Add donor path segments at unoccupied Harmonic endpoints, never rewrite it."""
from collections import Counter
from itertools import product
import numpy as np
from scipy.spatial import cKDTree

SCALE=np.array([1.625,.40625,.40625])
CONFIG=dict(anchor_um=1.,exclusion_um=3.,minimum_extension_edges=3)

def recover(reference,reference_edges,donor,donor_edges):
    nodes={k:dict(v) for k,v in reference.items()};edges=set(reference_edges)
    incoming=Counter(b for a,b in edges);outgoing=Counter(a for a,b in edges)
    xyz=lambda d,k:np.array([d[k][a] for a in ('z','y','x')])*SCALE
    mapped={};blocked=set();groups={}
    for k,v in reference.items():groups.setdefault(v['t'],[]).append(k)
    donor_frames={}
    for k,v in donor.items():donor_frames.setdefault(v['t'],[]).append(k)
    for t,ids in donor_frames.items():
        refs=groups.get(t,[])
        if not refs:continue
        a=np.array([xyz(donor,k) for k in ids]);b=np.array([xyz(reference,k) for k in refs])
        distance,index=cKDTree(b).query(a);_,reverse=cKDTree(a).query(b)
        for i,k in enumerate(ids):
            if distance[i]<=CONFIG['anchor_um'] and reverse[index[i]]==i:mapped[k]=refs[index[i]]
            elif distance[i]<=CONFIG['exclusion_um']:blocked.add(k)
    parent={};child={}
    for a,b in donor_edges:
        if a in child or b in parent:raise ValueError('Donor must have nonbranching paths')
        if donor[b]['t']!=donor[a]['t']+1:raise ValueError('Nonconsecutive donor edge')
        child[a]=b;parent[b]=a
    candidates=[];counts=Counter()
    def consume(run):
        anchors=[i for i,k in enumerate(run) if k in mapped]
        if not anchors:return
        intervals=list(zip(anchors[:-1],anchors[1:]))
        if anchors[0]>0:intervals.append((0,anchors[0]))
        if anchors[-1]<len(run)-1:intervals.append((anchors[-1],len(run)-1))
        for start,end in intervals:
            segment=run[start:end+1];attached=int(segment[0] in mapped)+int(segment[-1] in mapped)
            if len(segment)<2:continue
            if attached<2 and len(segment)-1<CONFIG['minimum_extension_edges']:
                counts['short_extensions_rejected']+=1;continue
            candidates.append((attached,segment))
    for first in sorted(set(child)-set(parent)):
        run=[];node=first
        while True:
            if node in blocked:consume(run);run=[]
            else:run.append(node)
            if node not in child:break
            node=child[node]
        consume(run)
    # New donor nodes cannot duplicate existing reference or accepted donor nodes.
    grid={};radius=CONFIG['exclusion_um'];offsets=list(product([-1,0,1],repeat=3))
    def cell(pos):return tuple(np.floor(pos/radius).astype(int))
    def register(node):
        pos=xyz(nodes,node);key=(nodes[node]['t'],*cell(pos));grid.setdefault(key,[]).append(pos)
    for node in nodes:register(node)
    next_id=max(nodes,default=-1)+1
    for attached,segment in sorted(candidates,key=lambda p:(-p[0],-len(p[1]),tuple(p[1]))):
        first,last=segment[0],segment[-1]
        if len(segment)==2 and first in mapped and last in mapped and (mapped[first],mapped[last]) in edges:
            counts['already_present']+=1;continue
        if first in mapped and outgoing[mapped[first]]>0:
            counts['occupied_endpoint_rejected']+=1;continue
        if last in mapped and incoming[mapped[last]]>0:
            counts['occupied_endpoint_rejected']+=1;continue
        new=[k for k in segment if k not in mapped];conflict=False
        for k in new:
            pos=xyz(donor,k);base=cell(pos);t=donor[k]['t']
            for offset in offsets:
                key=(t,*(base[j]+offset[j] for j in range(3)))
                if any(np.linalg.norm(pos-q)<=radius for q in grid.get(key,[])):conflict=True;break
            if conflict:break
        if conflict:counts['spatial_conflict_rejected']+=1;continue
        translated={k:mapped[k] for k in segment if k in mapped}
        for k in new:
            translated[k]=next_id;nodes[next_id]=dict(donor[k]);register(next_id);next_id+=1
        for a,b in zip(segment[:-1],segment[1:]):
            edge=(translated[a],translated[b]);edges.add(edge);outgoing[edge[0]]+=1;incoming[edge[1]]+=1
        counts['accepted_bridges' if attached==2 else 'accepted_extensions']+=1
    assert set(reference_edges)<=edges
    assert max(incoming.values(),default=0)<=1 and max(outgoing.values(),default=0)<=2
    assert {k for k,v in outgoing.items() if v==2}=={k for k,v in Counter(a for a,b in reference_edges).items() if v==2}
    for k,v in reference.items():assert nodes[k]==v
    return nodes,sorted(edges),dict(config=CONFIG,anchors=len(mapped),blocked_near_reference=len(blocked),
        proposed_segments=len(candidates),added_nodes=len(nodes)-len(reference),added_edges=len(edges)-len(reference_edges),
        counts=dict(counts),reference_edges_preserved=True,division_parents_preserved=True,annotations_used=False)
