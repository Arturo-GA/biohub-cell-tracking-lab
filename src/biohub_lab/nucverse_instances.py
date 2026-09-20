"""CPU gradient grouping adapted from the pinned MIT NucVerse3D predictor.

Retains its fixed 200-step momentum, seed threshold and minimum instance size;
stores only the final positions instead of the entire unused trajectory history.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

SCALE=np.array([1.625,.40625,.40625])

def centers_from_fields(fields):
    fields=np.asarray(fields)
    if fields.ndim!=4 or fields.shape[-1]!=4 or not np.isfinite(fields).all():raise ValueError('Invalid fields')
    shape=fields.shape[:3];initial=np.array(np.where(fields[...,0]>.5),dtype=np.int32)
    if not initial.shape[1]:return np.empty((0,3)),dict(foreground_voxels=0,instances=0)
    gradient=np.asarray(fields[...,1:],np.float64);vector=initial.astype(np.float64);change=np.zeros_like(vector)
    for _ in range(200):
        sampled=np.array([ndi.map_coordinates(gradient[...,a],vector,order=1,mode='constant',cval=0) for a in range(3)])
        change=.2*sampled+.8*change;vector+=change
    final=vector.astype(np.int32)
    valid=np.all((final>=0)&(final<np.array(shape)[:,None]),axis=0);final=final[:,valid];initial=initial[:,valid]
    votes=np.zeros(shape,np.int32);np.add.at(votes,tuple(final),1)
    smooth=ndi.gaussian_filter(votes,1)
    seeds,count=ndi.label(smooth>smooth.mean()+5*smooth.std(),structure=np.ones((3,3,3)))
    if not count:return np.empty((0,3)),dict(foreground_voxels=int(valid.size),in_bounds_voxels=int(valid.sum()),instances=0)
    positions=np.argwhere(seeds>0);labels=seeds[tuple(positions.T)]
    distance,index=cKDTree(positions).query(final.T,distance_upper_bound=5)
    matched=distance<5;assigned=labels[index[matched]];points=initial[:,matched]
    sizes=np.bincount(assigned,minlength=count+1);keep=np.flatnonzero(sizes>=20);keep=keep[keep>0]
    centers=np.array([np.bincount(assigned,weights=points[a],minlength=count+1)[keep]/sizes[keep] for a in range(3)]).T
    return centers,dict(foreground_voxels=int(valid.size),in_bounds_voxels=int(valid.sum()),seeds=int(count),instances=len(centers),assigned_voxels=int(matched.sum()))

def matched_truth(prediction,truth,max_distance=7.):
    prediction=np.asarray(prediction,float).reshape(-1,3);truth=np.asarray(truth,float).reshape(-1,3)
    if not len(truth) or not len(prediction):return set()
    distances=cdist(truth*SCALE,prediction*SCALE);penalty=(len(truth)+1)*(max_distance+1)
    cost=np.c_[np.where(distances<=max_distance,distances,2*penalty),np.full((len(truth),len(truth)),penalty)]
    rows,cols=linear_sum_assignment(cost)
    return set(int(r) for r,c in zip(rows,cols) if c<len(prediction) and distances[r,c]<=max_distance)
