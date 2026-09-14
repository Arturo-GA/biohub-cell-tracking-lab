"""Add disjoint missing-daughter links without deleting any baseline association."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch

from .mitosis_model import MitosisSpecialist
from .temporal_data import SCALE,TemporalConfig,load_volume,extract_patches


def validate_graph(coords,edges):
    coords=np.asarray(coords);edges=np.asarray(edges,np.int64).reshape(-1,2)
    if len(edges):
        if np.any(edges<0) or np.any(edges>=len(coords)): raise ValueError('Edge index out of bounds')
        if np.any(coords[edges[:,1],0]!=coords[edges[:,0],0]+1): raise ValueError('Nonconsecutive edge')
        if len(set(map(tuple,edges)))!=len(edges): raise ValueError('Duplicate baseline edge')
        if max(Counter(edges[:,0]).values())>2 or max(Counter(edges[:,1]).values())>1:
            raise ValueError('Invalid baseline lineage degree')
    return edges


def repair_candidates(coords,baseline,max_distance_um=20.,max_orphans=4):
    """A mother has one child; the proposed second child has no baseline parent."""
    coords=np.asarray(coords);baseline=validate_graph(coords,baseline)
    incoming=Counter(baseline[:,1]);outgoing={}
    for mother,child in baseline: outgoing.setdefault(int(mother),[]).append(int(child))
    triples=[]
    for t in np.unique(coords[:,0]):
        mothers=[int(m) for m in np.flatnonzero(coords[:,0]==t) if len(outgoing.get(int(m),[]))==1]
        orphans=np.array([n for n in np.flatnonzero(coords[:,0]==t+1) if incoming[n]==0],np.int64)
        if not mothers or not len(orphans): continue
        tree=cKDTree(coords[orphans,1:]*SCALE)
        for mother in mothers:
            child=outgoing[mother][0]
            if np.linalg.norm((coords[child,1:]-coords[mother,1:])*SCALE)>max_distance_um: continue
            indices=tree.query_ball_point(coords[mother,1:]*SCALE,max_distance_um)
            ordered=sorted(indices,key=lambda i:(float(np.linalg.norm((coords[orphans[i],1:]-coords[mother,1:])*SCALE)),int(orphans[i])))[:max_orphans]
            # Child A remains the baseline child. The neural score is daughter symmetric.
            triples.extend((mother,child,int(orphans[i])) for i in ordered)
    return np.asarray(triples,np.int64).reshape(-1,3)


def select_repairs(coords,baseline,triples,scores,threshold):
    """Maximum weight bipartite assignment, with abstention, separately per frame."""
    coords=np.asarray(coords);baseline=validate_graph(coords,baseline)
    triples=np.asarray(triples,np.int64).reshape(-1,3);scores=np.asarray(scores,np.float64)
    if len(triples)!=len(scores) or not np.isfinite(scores).all(): raise ValueError('Invalid repair scores')
    if threshold is not None and not np.isfinite(threshold): raise ValueError('Invalid development threshold')
    base=set(map(tuple,baseline));incoming=Counter(baseline[:,1]);outgoing=Counter(baseline[:,0])
    pairs=set()
    for mother,child,orphan in triples:
        if min(mother,child,orphan)<0 or max(mother,child,orphan)>=len(coords): raise ValueError('Invalid candidate index')
        if (mother,child) not in base or outgoing[mother]!=1 or incoming[orphan]!=0:
            raise ValueError('Repair would overwrite a baseline relationship')
        if child==orphan or coords[orphan,0]!=coords[mother,0]+1: raise ValueError('Invalid missing daughter')
        if (mother,orphan) in pairs: raise ValueError('Duplicate repair candidate')
        pairs.add((mother,orphan))
    eligible=np.zeros(len(scores),bool) if threshold is None else scores>=threshold
    selected=[]
    for t in np.unique(coords[triples[eligible,0],0]):
        refs=np.flatnonzero(eligible&(coords[triples[:,0],0]==t))
        mothers=sorted(set(triples[refs,0]));orphans=sorted(set(triples[refs,2]))
        m={v:i for i,v in enumerate(mothers)};o={v:i for i,v in enumerate(orphans)}
        weights=np.full((len(m),len(o)+len(m)),-1e12,np.float64)
        weights[:,len(o):]=0. # Each mother may abstain through a dummy column.
        lookup={}
        for ref in refs:
            mother,_,orphan=triples[ref];i,j=m[mother],o[orphan]
            weights[i,j]=max(float(scores[ref]-threshold),1e-6);lookup[(i,j)]=int(ref)
        rows,cols=linear_sum_assignment(-weights)
        selected.extend(lookup[(i,j)] for i,j in zip(rows,cols) if (i,j) in lookup)
    selected=np.asarray(sorted(selected),np.int64)
    additions=triples[selected][:,[0,2]]
    result=np.asarray(sorted(base|set(map(tuple,additions))),np.int64).reshape(-1,2)
    validate_graph(coords,result)
    if not base<=set(map(tuple,result)): raise ValueError('A baseline edge was deleted')
    return result,dict(candidates=len(triples),above_dev_threshold=int(eligible.sum()),added_edges=len(additions),
        preserved_baseline_edges=len(base),removed_baseline_edges=0,threshold=threshold),selected


def load_specialist(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=MitosisSpecialist(TemporalConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model,checkpoint


@torch.inference_mode()
def repair_video(model,checkpoint,coords,baseline,image_path,device):
    coords=np.asarray(coords,np.float32)
    triples=repair_candidates(coords,baseline,max_distance_um=model.config.max_distance_um)
    scores=[]
    if len(triples):
        volume=load_volume(image_path,model.config)
        nodes=np.unique(triples);lookup=np.full(len(coords),-1,np.int64);lookup[nodes]=np.arange(len(nodes))
        patches=extract_patches(volume,coords[nodes],model.config)
        del volume
        for start in range(0,len(triples),48):
            tri=triples[start:start+48]
            patch=torch.as_tensor(patches[lookup[tri]],device=device)
            delta=torch.as_tensor((coords[tri[:,1:],1:]-coords[tri[:,0],1:][:,None])*SCALE,device=device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                scores.extend(model(patch,delta).float().cpu().tolist())
    scores=np.asarray(scores,np.float32)
    result,stats,selected=select_repairs(coords,baseline,triples,scores,checkpoint['threshold'])
    return result,stats,dict(triples=triples,scores=scores,selected_repairs=selected)
