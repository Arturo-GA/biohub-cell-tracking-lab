"""Sparse-label examples and raw five-frame image patches for temporal learning."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

SCALE=np.array([1.625,.40625,.40625],np.float32)


@dataclass(frozen=True)
class TemporalConfig:
    context: int=5
    patch_shape: tuple=(9,17,17)
    downsample: tuple=(1,4,4)
    max_parents: int=12
    max_distance_um: float=20.
    distractor_exclusion_um: float=5.
    embedding: int=64
    seed: int=20260914


def load_gt(path):
    import zarr
    g=zarr.open_group(str(path),mode='r')
    ids=np.asarray(g['nodes/ids'][:])
    coords=np.column_stack([g[f'nodes/props/{k}/values'][:] for k in ('t','z','y','x')]).astype(np.float32)
    lookup={int(v):i for i,v in enumerate(ids)}
    edges=np.array([[lookup[int(s)],lookup[int(t)]] for s,t in g['edges/ids'][:]],dtype=np.int64).reshape(-1,2)
    if len(edges): edges=edges[coords[edges[:,1],0]==coords[edges[:,0],0]+1]
    return coords,edges


def parent_candidates(coords,config=TemporalConfig()):
    coords=np.asarray(coords)
    parents=np.full((len(coords),config.max_parents),-1,np.int64)
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t-1); b=np.flatnonzero(coords[:,0]==t)
        if not len(a) or not len(b): continue
        dist,idx=cKDTree(coords[a,1:]*SCALE).query(coords[b,1:]*SCALE,
            k=config.max_parents,distance_upper_bound=config.max_distance_um)
        dist,idx=dist.reshape(len(b),-1),idx.reshape(len(b),-1)
        valid=np.isfinite(dist)
        local=np.full(idx.shape,-1,np.int64); local[valid]=a[idx[valid]]
        parents[b]=local
    return parents


def examples(coords,edges,config=TemporalConfig()):
    parents=parent_candidates(coords,config)
    # Only annotated edges establish a supervised parent. Unknown targets have no loss.
    true_parent=np.full(len(coords),-1,np.int64)
    children={}
    for s,t in edges:
        if true_parent[t]>=0: raise ValueError('GT target has multiple parents')
        true_parent[t]=s
        children.setdefault(int(s),[]).append(int(t))
    targets=np.flatnonzero(true_parent>=0)
    labels=np.full(len(coords),-1,np.int64)
    for t in targets:
        hit=np.flatnonzero(parents[t]==true_parent[t])
        if len(hit): labels[t]=hit[0]
    triples=[]; triple_labels=[]
    rng=np.random.default_rng(config.seed)
    # A negative pair requires at least one daughter whose KNOWN parent differs.
    # Single annotated children do not label their parent as a non-dividing cell.
    for s,kids in sorted(children.items()):
        if len(kids)>2: raise ValueError('Non-binary annotated lineage')
        if len(kids)==2:
            triples.append((s,*sorted(kids))); triple_labels.append(1)
        pool=targets[(coords[targets,0]==coords[s,0]+1)&(true_parent[targets]!=s)]
        if len(pool):
            dist=np.linalg.norm((coords[pool,1:]-coords[s,1:])*SCALE,axis=1)
            pool=pool[np.argsort(dist)[:4]]
            pool=pool[np.linalg.norm((coords[pool,1:]-coords[s,1:])*SCALE,axis=1)<config.max_distance_um]
            for other in pool:
                triples.append((s,*sorted((int(rng.choice(kids)),int(other))))); triple_labels.append(0)
    return dict(parents=parents,labels=labels,targets=targets,true_parent=true_parent,
        triples=np.asarray(triples,np.int64).reshape(-1,3),triple_labels=np.asarray(triple_labels,np.float32))


def extract_patches(volume,coords,config=TemporalConfig(),output=None):
    """Volume is uint8 TZYX at config.downsample; no annotations used in extraction."""
    shape=(len(coords),config.context,*config.patch_shape)
    out=np.empty(shape,np.uint8) if output is None else output
    half=np.asarray(config.patch_shape)//2
    ds=np.asarray(config.downsample)
    for i,(t,*xyz) in enumerate(coords):
        center=np.rint(np.asarray(xyz)/ds).astype(int)
        lo=center-half; hi=lo+np.asarray(config.patch_shape)
        src_lo=np.maximum(lo,0); src_hi=np.minimum(hi,volume.shape[1:])
        out[i]=0
        if np.any(src_hi<=src_lo): continue
        dst_lo=src_lo-lo; dst_hi=dst_lo+src_hi-src_lo
        for j,offset in enumerate(range(-config.context//2+1,config.context//2+1)):
            frame=int(np.clip(int(t)+offset,0,len(volume)-1))
            out[(i,j,*[slice(a,b) for a,b in zip(dst_lo,dst_hi)])]=volume[(frame,*[slice(a,b) for a,b in zip(src_lo,src_hi)])]
    return out


def load_volume(path,config=TemporalConfig()):
    import zarr
    g=zarr.open_group(str(path),mode='r'); image=g['0']
    ds=config.downsample
    shape=(image.shape[0],*[(n+d-1)//d for n,d in zip(image.shape[1:],ds)])
    volume=np.empty(shape,np.uint8)
    # Normalize each frame from its own image, independent of labels and other videos.
    for t in range(image.shape[0]):
        frame=np.asarray(image[t,::ds[0],::ds[1],::ds[2]],np.float32)
        low,high=np.quantile(frame[::2,::2,::2],[.01,.998])
        volume[t]=np.rint(np.clip((frame-low)/max(float(high-low),1.),0,1)*255).astype(np.uint8)
    return volume


def add_distractors(volume,coords,config=TemporalConfig()):
    extra=[]; ds=np.asarray(config.downsample)
    for t in np.unique(coords[:,0]).astype(int):
        gt=coords[coords[:,0]==t,1:]
        sm=gaussian_filter(volume[t].astype(np.float32),.8)
        dog=sm-gaussian_filter(volume[t].astype(np.float32),1.8)
        pts=np.argwhere((dog==maximum_filter(dog,size=3))&(dog>3.)&(sm>20.))
        if not len(pts): continue
        xyz=pts*ds
        dist,_=cKDTree(gt*SCALE).query(xyz*SCALE)
        valid=(dist>config.distractor_exclusion_um)&(dist<config.max_distance_um)
        pts,xyz=pts[valid],xyz[valid]
        if not len(pts): continue
        score=dog[tuple(pts.T)]
        keep=np.argsort(-score,kind='stable')[:len(gt)]
        extra.extend((t,*v) for v in xyz[keep])
    return np.concatenate([coords,np.asarray(extra,np.float32).reshape(-1,4)])


def prepare_video(image_path,output,config=TemporalConfig()):
    import time
    start=time.monotonic(); image_path=Path(image_path); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    coords,edges=load_gt(image_path.with_suffix('.geff'))
    gt_nodes=len(coords)
    config_hash=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()+coords.tobytes()+edges.tobytes()).hexdigest()
    receipt=output/'receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['config_hash']!=config_hash: raise ValueError('Cache configuration changed')
        return r
    volume=load_volume(image_path,config)
    coords=add_distractors(volume,coords,config)
    shape=(len(coords),config.context,*config.patch_shape)
    patches=np.lib.format.open_memmap(output/'patches.npy',mode='w+',dtype=np.uint8,shape=shape)
    extract_patches(volume,coords,config,patches); patches.flush(); del patches,volume
    ex=examples(coords,edges,config)
    np.savez(output/'graph.npz',coords=coords,edges=edges,**ex)
    r=dict(name=image_path.stem,group=image_path.stem.split('_')[0],gt_nodes=gt_nodes,nodes=len(coords),
        gt_edges=len(edges),covered_edges=int((ex['labels']>=0).sum()),
        division_positive=int(ex['triple_labels'].sum()),division_negative=int((ex['triple_labels']==0).sum()),
        seconds=time.monotonic()-start,config_hash=config_hash,config=asdict(config))
    receipt.write_text(json.dumps(r,indent=2))
    return r
