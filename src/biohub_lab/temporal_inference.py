"""Score dense detections and optimize singleton/division hypotheses jointly."""
from itertools import combinations
import math
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.special import expit, logit
import torch
from .temporal_data import SCALE, TemporalConfig, extract_patches, load_volume, parent_candidates
from .temporal_model import TemporalLinker


def solve_hypotheses(coords,edges,gains,triples,triple_bonuses,time_limit=10.):
    """Set packing: each parent picks one event; each daughter has at most one parent.

    No binary division can be assembled from unrelated high-probability edges.
    Both daughters must belong to one explicitly scored triple hypothesis.
    """
    coords=np.asarray(coords); edges=np.asarray(edges,np.int64).reshape(-1,2)
    triples=np.asarray(triples,np.int64).reshape(-1,3)
    gains=np.asarray(gains,float); bonuses=np.asarray(triple_bonuses,float)
    if len(edges)!=len(gains) or len(triples)!=len(bonuses): raise ValueError('Score shape mismatch')
    if not np.isfinite(gains).all() or not np.isfinite(bonuses).all(): raise ValueError('Invalid gains')
    if len(edges) and (np.any(edges<0) or np.any(edges>=len(coords)) or
        np.any(coords[edges[:,1],0]!=coords[edges[:,0],0]+1)): raise ValueError('Nonconsecutive edge')
    if len(np.unique(edges,axis=0))!=len(edges): raise ValueError('Duplicate edge')
    edge_gain={tuple(e):w for e,w in zip(edges,gains)}
    for s,a,b in triples:
        if a==b or (s,a) not in edge_gain or (s,b) not in edge_gain: raise ValueError('Invalid division hypothesis')
    selected=[]; states=[]
    for t in np.unique(coords[:,0]):
        options=[]; values=[]
        for i in np.flatnonzero(coords[edges[:,0],0]==t):
            s,a=edges[i]
            if gains[i]>0: options.append((int(s),int(a))); values.append(gains[i])
        for i in np.flatnonzero(coords[triples[:,0],0]==t):
            s,a,b=triples[i]
            value=edge_gain[(s,a)]+edge_gain[(s,b)]+bonuses[i]
            if value>0: options.append((int(s),int(a),int(b))); values.append(value)
        if not options: continue
        resources=sorted({('p',o[0]) for o in options}|{('d',v) for o in options for v in o[1:]})
        lookup={r:i for i,r in enumerate(resources)}
        rows=[]; cols=[]
        for j,o in enumerate(options):
            for r in [('p',o[0])]+[('d',v) for v in o[1:]]:
                rows.append(lookup[r]); cols.append(j)
        matrix=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(resources),len(options))).tocsc()
        solution=milp(-np.array(values),integrality=np.ones(len(options)),bounds=Bounds(0,1),
            constraints=LinearConstraint(matrix,0,1),options={'time_limit':time_limit,'mip_rel_gap':.001})
        if solution.x is None: raise RuntimeError(f'No feasible lineage for frame {t}: {solution.message}')
        x=np.rint(solution.x)
        if np.max(np.abs(x-solution.x))>1e-5 or np.any(matrix@x>1.00001):
            raise RuntimeError('Solver returned infeasible lineage')
        for j in np.flatnonzero(x):
            option=options[j]; selected.extend((option[0],d) for d in option[1:])
        states.append(dict(t=int(t),status=int(solution.status),gap=float(solution.mip_gap),options=len(options)))
    return np.asarray(sorted(selected),np.int64).reshape(-1,2),states


def load_model(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    config=TemporalConfig(**checkpoint['config'])
    model=TemporalLinker(config).to(device).eval(); model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint


@torch.inference_mode()
def score_video(model,checkpoint,coords,image_path,device):
    config=model.config; coords=np.asarray(coords,np.float32)
    volume=load_volume(image_path,config)
    # Encode each detected node once; five frames provide motion/appearance evidence.
    features=[]
    for start in range(0,len(coords),256):
        patch=extract_patches(volume,coords[start:start+256],config)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            features.append(model.encode(torch.as_tensor(patch,device=device)).float())
    del volume
    h=torch.cat(features); parents=parent_candidates(coords,config)
    probabilities=np.zeros((len(coords),config.max_parents+1),np.float32)
    for start in range(0,len(coords),256):
        ids=np.arange(start,min(start+256,len(coords))); src=parents[ids]; valid=src>=0
        delta=(coords[src.clip(0),1:]-coords[ids,1:][:,None])*SCALE
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            logits=model.parents(h,torch.as_tensor(src.clip(0),device=device),torch.as_tensor(ids,device=device),
                torch.as_tensor(delta,device=device),torch.as_tensor(valid,device=device))
        # Orphans trained by removing 10% of known parents. Keep that explicit prior;
        # no leaderboard-driven calibration is performed here.
        probabilities[ids]=logits.float().softmax(-1).cpu().numpy()
    target,slot=np.where(parents>=0)
    edges=np.column_stack([parents[target,slot],target])
    p=probabilities[target,slot]; orphan=probabilities[target,-1]
    gains=np.log(np.clip(p,1e-7,1))-np.log(np.clip(orphan,1e-7,1))
    children={}
    for (s,t),prob in zip(edges,p): children.setdefault(int(s),[]).append((float(prob),int(t)))
    triples=[]
    for s,items in children.items():
        kids=[t for prob,t in sorted(items,reverse=True)[:4] if prob>=.05]
        triples.extend((s,*sorted((a,b))) for a,b in combinations(kids,2))
    triples=np.asarray(triples,np.int64).reshape(-1,3); scores=[]
    prior=float(np.clip(checkpoint['division_prior'],1e-6,1-1e-6))
    correction=math.log(prior/(1-prior))
    for start in range(0,len(triples),512):
        part=triples[start:start+512]
        delta=(coords[part[:,1:],1:]-coords[part[:,0],1:][:,None])*SCALE
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            logits=model.divisions(h,torch.as_tensor(part,device=device),torch.as_tensor(delta,device=device))
        scores.extend(logits.float().cpu().numpy()+correction)
    scores=np.asarray(scores,np.float32)
    accepted=scores>=0. # Calibrated division evidence must favor a real fork.
    selected,solver=solve_hypotheses(coords,edges,gains,triples[accepted],scores[accepted])
    return selected,dict(nodes=len(coords),candidate_edges=len(edges),division_hypotheses=len(triples),
        accepted_division_hypotheses=int(accepted.sum()),selected_edges=len(selected),solver=solver,
        division_prior=prior),dict(edges=edges,probabilities=p,orphan=probabilities[:,-1],
        triples=triples,division_logits=scores)
