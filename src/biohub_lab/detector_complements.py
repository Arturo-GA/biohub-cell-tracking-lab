"""E045 fixed complementary mechanisms; no label-dependent prediction."""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def peaks(volume,budget=512):
    xyz=np.argwhere(volume==maximum_filter(volume,3))
    return xyz[np.argsort(-volume[tuple(xyz.T)],kind='stable')[:budget]]

def refine(anchors,targets,radius_um=3):
    """One-to-one local snap; preserve anchors if simultaneous snaps collide."""
    anchors=np.asarray(anchors);targets=np.asarray(targets)
    if not len(anchors) or not len(targets):return anchors.copy()
    _,inverse,counts=np.unique(anchors,axis=0,return_inverse=True,return_counts=True)
    movable=counts[inverse]==1
    if not np.all(movable):
        # Harmonic may contain different IDs at the same rounded coordinate.
        # Preserve those existing nodes; never create another at their position.
        blocked=anchors[~movable]
        available=targets[~np.any(np.all(targets[:,None,:]==blocked[None,:,:],axis=2),axis=1)]
        result=anchors.copy();result[movable]=refine(anchors[movable],available,radius_um)
        return result
    distance=cdist(anchors,targets)*1.625;penalty=(len(anchors)+1)*(radius_um+1)
    i,j=linear_sum_assignment(np.c_[np.where(distance<=radius_um,distance,2*penalty),np.full((len(anchors),len(anchors)),penalty)])
    result=anchors.copy()
    for a,b in zip(i,j):
        if b<len(targets) and distance[a,b]<=radius_um:result[a]=targets[b]
    # Reverting a collision can cause another one at an original anchor; resolve to a fixed point.
    while True:
        _,inverse,counts=np.unique(result,axis=0,return_inverse=True,return_counts=True)
        bad=(counts[inverse]>1)&np.any(result!=anchors,axis=1)
        if not np.any(bad):break
        result[bad]=anchors[bad]
    assert len(np.unique(result,axis=0))==len(result)
    return result

def standardized(volume):
    v=np.asarray(volume,np.float64);lo,hi=np.percentile(v,[10,90])
    return np.clip((v-np.median(v))/max(hi-lo,1e-6),-5,5)

def complement(dense,static,budget):
    keep=list(dense[:3*budget//4])
    for p in static:
        if not keep or np.min(np.linalg.norm(np.asarray(keep)-p,axis=1))*1.625>3:keep.append(p)
        if len(keep)==budget:break
    if len(keep)<budget:
        for p in dense[3*budget//4:]:
            if not any(np.array_equal(p,q) for q in keep):keep.append(p)
            if len(keep)==budget:break
    return np.asarray(keep)

def candidates(logits,static_score,temporal_score,image_peaks,static,temporal,budget):
    dense=peaks(logits)
    ds=standardized(logits);ss=standardized(np.log1p(static_score));ts=standardized(np.log1p(temporal_score))
    return dict(dense=dense[:budget],static=static[:budget],temporal=temporal[:budget],image_peaks=image_peaks[:budget],
        static_dense_refine=refine(static[:budget],dense),temporal_dense_refine=refine(temporal[:budget],dense),
        image_dense_refine=refine(image_peaks[:budget],dense),
        spatial_consensus=peaks(ds+ss,budget),spatiotemporal_consensus=peaks(ds+(ss+ts)/2,budget),
        dense_novel_static=complement(dense,static,budget))
