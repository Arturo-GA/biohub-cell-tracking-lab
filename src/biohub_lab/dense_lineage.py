"""Procedural, fully labelled 3D movies for acquisition-safe pretraining.

Original renderer; inspired by synthetic cell-tracking literature, not a port of
SynCellFactory/ControlNet or of the public Biohub dataset generator.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates

from .temporal_data import TemporalConfig,SCALE,examples,extract_patches


def template_bank(store,names,held_group,limit=256,seed=20260914):
    """Extract central, tapered nuclear textures from TRAIN videos only."""
    if not names or any(n.split('_')[0]==held_group for n in names):
        raise ValueError('Held-out embryo must not contribute synthetic appearance')
    rng=np.random.default_rng(seed); bank=[]; sources=[]
    for name in sorted(names):
        g=store.graphs[name]
        ids=np.unique(g['edges'].ravel())
        if not len(ids): continue
        take=rng.choice(ids,min(4,len(ids)),replace=False)
        for node in take:
            patch=np.asarray(store.patches[name][int(node),2],np.float32)
            center=(np.asarray(patch.shape)-1)/2
            grid=np.indices(patch.shape).astype(np.float32)
            radius=np.array([3.,5.,5.])
            mask=np.exp(-.5*sum(((grid[k]-center[k])/radius[k])**4 for k in range(3)))
            patch=np.maximum(patch-np.quantile(patch,.25),0)*mask
            high=float(np.quantile(patch,.99))
            if high<5: continue
            bank.append(np.clip(patch/high,0,1));sources.append(dict(video=name,node=int(node)))
    if not bank: raise ValueError('No usable training textures')
    selected=rng.permutation(len(bank))[:limit]
    return np.stack([bank[i] for i in selected]),[sources[i] for i in selected]


def render_scene(seed,bank=None,shape=(17,33,33),cells=10,division_probability=.22,force_crossing=False,frames=5):
    """Every rendered cell has a node and all parent edges are known.

    Daughters inherit their mother's texture and conserve approximate integrated
    fluorescence. Independent nearby cells continue moving through mitosis frames.
    Coordinates are exported in Biohub's original anisotropic voxel convention.
    """
    rng=np.random.default_rng(seed); shape=np.asarray(shape); T=frames
    if T!=5 and T<7: raise ValueError('Use five frames or at least seven for full event context')
    division_range=(1,4) if T==5 else (3,T-2)
    if np.any(shape<13): raise ValueError('Scene too small for trajectories')
    volume=np.zeros((T,*shape),np.float32); coords=[]; edges=[]; prev={}; events=[]
    margin=np.array([4.,7.,7.]); center=(shape-1)/2
    scene_velocity=rng.normal(0,.22,3)
    states=[]
    for identity in range(cells):
        pos=rng.uniform(margin,shape-1-margin)
        velocity=scene_velocity+rng.normal(0,.18,3)
        texture=None if bank is None else bank[int(rng.integers(len(bank)))]
        radius=rng.uniform(1.05,2.25,3)
        direction=rng.normal(size=3); direction/=np.linalg.norm(direction)
        states.append(dict(identity=identity,pos=pos,velocity=velocity,radius=radius,
            texture=texture,amplitude=rng.uniform(45,150),division=(int(rng.integers(*division_range)) if rng.random()<division_probability else -1),
            direction=direction,parent=None))
    if force_crossing:
        if cells<2: raise ValueError('A crossing needs two independent cells')
        for i,sign in enumerate((-1,1)):
            states[i].update(pos=center+np.array([sign*.9,0,sign*2.]),
                velocity=np.array([0,0,-sign*.7]),division=-1)
    next_id=cells
    for t in range(T):
        active=[]
        for state in states:
            state=dict(state); state['pos']=state['pos']+state['velocity']+rng.normal(0,.08,3)
            state['pos']=np.clip(state['pos'],margin,shape-1-margin)
            if state['division']==t:
                pair=[]
                for sign in (-1,1):
                    child=dict(state,identity=next_id,parent=state['identity'],division=-1)
                    child['pos']=state['pos']+sign*state['direction']*rng.uniform(1.3,1.9)
                    child['velocity']=state['velocity']+sign*state['direction']*.30
                    child['radius']=state['radius']*np.cbrt(.5)
                    active.append(child);pair.append(next_id);next_id+=1
                events.append(dict(t=t-1,mother=state['identity'],daughters=pair))
            else: active.append(state)
        current={}
        for state in active:
            pos=state['pos']; rad=state['radius']; identity=state['identity']
            # A small bilobed deformation immediately before division, with varying direction.
            dividing_next=state['division']==t+1
            extent=np.ceil(rad*3).astype(int)+1
            lo=np.maximum(np.floor(pos).astype(int)-extent,0); hi=np.minimum(np.floor(pos).astype(int)+extent+1,shape)
            grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij')).astype(np.float32)
            relative=grid-pos[:,None,None,None]
            envelope=np.exp(-.5*np.sum((relative/rad[:,None,None,None])**2,axis=0))
            if dividing_next:
                axis=state['direction'][:,None,None,None]*rng.uniform(.3,.8)
                envelope=.5*(np.exp(-.5*np.sum(((relative-axis)/rad[:,None,None,None])**2,axis=0))+
                    np.exp(-.5*np.sum(((relative+axis)/rad[:,None,None,None])**2,axis=0)))
            if state['texture'] is not None:
                texture=state['texture']; tc=(np.asarray(texture.shape)-1)/2
                sample=relative/rad[:,None,None,None]*2+tc[:,None,None,None]
                signal=map_coordinates(texture,sample,order=1,mode='constant',cval=0)
                envelope*=.4+.6*signal
            volume[(t,*[slice(a,b) for a,b in zip(lo,hi)])]+=state['amplitude']*envelope
            node=len(coords); current[identity]=node
            # Jitter simulates detector localization without changing event labels.
            observed=np.clip(pos+rng.normal(0,.18,3),0,shape-1)
            coords.append((t,*(observed*np.array([1,4,4]))))
            parent=identity if identity in prev else state['parent']
            if parent in prev: edges.append((prev[parent],node))
        states=active;prev=current
    background=gaussian_filter(rng.normal(0,1,tuple(shape)),2)
    background=(background-background.min())/(np.ptp(background)+1e-6)*rng.uniform(4,25)
    for t in range(T):
        frame=gaussian_filter(volume[t],rng.uniform(.25,.65))
        frame=frame*rng.uniform(.85,1.15)+background+rng.uniform(2,12)
        gain=rng.uniform(.5,3.)
        frame=rng.poisson(np.maximum(frame,0)*gain)/gain+rng.normal(0,rng.uniform(.5,4),tuple(shape))
        low,high=np.quantile(frame,[.01,.998])
        volume[t]=np.clip((frame-low)/max(high-low,1),0,1)*255
    return np.rint(volume).astype(np.uint8),np.asarray(coords,np.float32),np.asarray(edges,np.int64).reshape(-1,2),events


def dense_examples(coords,edges,config=TemporalConfig()):
    """Every same-time competing pair is known here; retain difficult near pairs."""
    from itertools import combinations
    ex=examples(coords,edges,config); children={}
    for s,t in edges: children.setdefault(int(s),[]).append(int(t))
    triples=[]; labels=[]
    for s,kids in sorted(children.items()):
        future=np.flatnonzero(coords[:,0]==coords[s,0]+1)
        distance=np.linalg.norm((coords[future,1:]-coords[s,1:])*SCALE,axis=1)
        pool=set(future[np.argsort(distance)[:4]][np.sort(distance)[:4]<config.max_distance_um])|set(kids)
        for a,b in combinations(sorted(pool),2):
            triples.append((s,a,b));labels.append(int(set(kids)=={a,b}))
    ex.update(triples=np.asarray(triples,np.int64).reshape(-1,3),triple_labels=np.asarray(labels,np.float32))
    return ex


def build_cache(store,split,root,scenes=2048,seed=7302026,frames=5):
    root=Path(root);root.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    bank,sources=template_bank(store,split['train'],split['held_group'])
    config=TemporalConfig(); rows=[]; all_events=0
    rng=np.random.default_rng(seed)
    scene_seeds=rng.integers(0,2**31-1,size=scenes)
    for i,s in enumerate(scene_seeds):
        crossing=bool(rng.random()<.25)
        volume,coords,edges,events=render_scene(int(s),bank,cells=int(rng.integers(7,14)),force_crossing=crossing,frames=frames)
        ex=dense_examples(coords,edges,config); name=f'synthetic_{i:05d}'; folder=root/name;folder.mkdir(exist_ok=True)
        np.save(folder/'patches.npy',extract_patches(volume,coords,config))
        np.savez(folder/'graph.npz',coords=coords,edges=edges,**ex)
        row=dict(name=name,seed=int(s),nodes=len(coords),edges=len(edges),divisions=len(events),forced_crossing=crossing,
            positive_pairs=int(ex['triple_labels'].sum()),negative_pairs=int((ex['triple_labels']==0).sum()))
        rows.append(row);all_events+=len(events)
        if i<4: np.savez_compressed(root/f'preview_{i}.npz',volume=volume,coords=coords,edges=edges)
        if (i+1)%128==0: print('DENSE_CACHE',split['held_group'],i+1,scenes,'divisions',all_events,flush=True)
    receipt=dict(complete=True,videos=rows,config=asdict(config),template_sources=sources,
        template_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),held_group=split['held_group'],
        training_videos=split['train'],dev_videos=split['dev'],seed=seed,seconds=time.monotonic()-start,
        generated_divisions=all_events,frames=frames,scope='Fully labelled procedural movies; train-embryo textures only; no claim of photorealistic validation.')
    (root/'cache_manifest.json').write_text(json.dumps(receipt,indent=2))
    return receipt
