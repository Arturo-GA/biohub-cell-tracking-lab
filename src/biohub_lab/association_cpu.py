"""CPU scoring and fixed structured decoding for the trained E014 heads."""
import numpy as np
import torch
from .association_rank import node_features, SCALE
from .event_data import candidates


def calibrated_threshold(scores, labels, beta):
    scores=np.asarray(scores);labels=np.asarray(labels)
    if not np.isfinite(scores).all() or not np.isin(labels,(0,1)).all() or not 0<labels.sum()<len(labels):
        raise ValueError('Calibration needs finite scores and both known classes')
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[ends]
    fp=ends+1-tp;fn=y.sum()-tp
    metric=(1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)
    # Descending thresholds: argmax breaks ties toward the conservative one.
    best=int(np.argmax(metric))
    return dict(logit=float(s[ends[best]]),f_beta=float(metric[best]),beta=beta,
        tp=int(tp[best]),fp=int(fp[best]),fn=int(fn[best]),positives=int(y.sum()),negatives=int(len(y)-y.sum()))


def top_per_mother(indices,gains,count):
    if not len(indices):return indices,gains
    order=np.lexsort((np.arange(len(gains)),-gains,indices[:,0]));m=indices[order,0]
    starts=np.maximum.accumulate(np.where(np.r_[True,m[1:]!=m[:-1]],np.arange(len(order)),0))
    keep=order[np.arange(len(order))-starts<count]
    return indices[keep],gains[keep]


@torch.inference_mode()
def score_events(graph,visual,models,thresholds):
    if any(next(m.parameters()).device.type!='cpu' for m in models.values()):
        raise ValueError('This workflow is CPU-only')
    nodes=node_features(graph,visual);positions=graph['coords'][:,1:].astype(np.float32)*SCALE/20.
    def scores(head,indices):
        values=[]
        for first in range(0,len(indices),4096):
            ix=indices[first:first+4096]
            values.append(models[head](torch.from_numpy(nodes[ix]),torch.from_numpy(positions[ix])).numpy())
        return np.concatenate(values) if values else np.empty(0,np.float32)
    all_edges=[];all_gains=[];all_pairs=[];all_pair_gains=[];total_edges=total_pairs=0
    for first in range(0,len(nodes),256):
        mothers=np.arange(first,min(first+256,len(nodes)))
        edges,triples=candidates(graph,mothers);edge_scores=scores('edge',edges);pair_scores=scores('division',triples)
        total_edges+=len(edges);total_pairs+=len(triples)
        values=np.full(graph['daughters'][mothers].shape,-np.inf,np.float32)
        rr,cc=np.nonzero(graph['daughters'][mothers]>=0);values[rr,cc]=edge_scores
        if len(triples):
            allowed=np.unpackbits(graph['pair_bits'][mothers],axis=1,bitorder='little')[:,:len(graph['pair_slots'])]
            rows,slots=np.nonzero(allowed);pair=graph['pair_slots'][slots]
            gain=.5*(values[rows,pair[:,0]]+values[rows,pair[:,1]])-thresholds['edge']['logit']
            gain+=2*(pair_scores-thresholds['division']['logit'])
            valid=pair_scores>=thresholds['division']['logit']
            ix,g=top_per_mother(triples[valid],gain[valid],2);all_pairs.append(ix);all_pair_gains.append(g)
        if len(edges):
            valid=(graph['continuation'][edges[:,0]]==edges[:,1,None]).any(1)
            ix,g=top_per_mother(edges[valid],edge_scores[valid]-thresholds['edge']['logit'],4)
            all_edges.append(ix);all_gains.append(g)
    if total_pairs!=int(graph['pair_counts'].sum()):raise ValueError('Incomplete division scoring')
    def cat(parts,width=None):
        return np.concatenate(parts) if parts else np.empty((0,width),np.int64) if width else np.empty(0,np.float32)
    return dict(edges=cat(all_edges,2),edge_gains=cat(all_gains),triples=cat(all_pairs,3),
        triple_gains=cat(all_pair_gains),quality=np.full(len(nodes),.5,np.float32)),dict(
        scored_edges=total_edges,scored_pairs=total_pairs,quality='Fixed neutral 0.5; E014 has no quality head',
        pruning='Top four continuations and top two threshold-passing divisions per mother')
