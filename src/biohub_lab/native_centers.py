"""Native-image centroid corrections with temporal and identity safeguards."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

SCALE=np.array([1.625,.40625,.40625])

def image_offsets(frame,points):
    offsets=np.stack(np.meshgrid(np.arange(-2,3),np.arange(-6,7),np.arange(-6,7),indexing='ij'),-1).reshape(-1,3)
    prior=np.exp(-.5*np.sum((offsets*SCALE/1.2)**2,axis=1));result=[]
    for start in range(0,len(points),256):
        p=points[start:start+256];at=p[:,None,:]+offsets[None];inside=((at>=0)&(at<np.asarray(frame.shape))).all(2)
        at=np.clip(at,0,np.asarray(frame.shape)-1);v=np.asarray(frame[at[:,:,0],at[:,:,1],at[:,:,2]],float)
        background=np.percentile(v,20,axis=1);signal=np.maximum(v-background[:,None],0)
        # Squared contrast favors the nucleus interior over diffuse surroundings.
        w=signal**2*prior[None]*inside;mass=w.sum(1)
        delta=np.einsum('nk,kd->nd',w,offsets)/np.maximum(mass[:,None],1e-12)
        norm=np.linalg.norm(delta*SCALE,axis=1);delta*=np.minimum(1,1.2/np.maximum(norm,1e-12))[:,None]
        result.append(delta)
    return np.concatenate(result) if result else np.empty((0,3))

def temporal_offsets(coords,edges,delta):
    incoming=Counter(int(b) for a,b in edges);outgoing=Counter(int(a) for a,b in edges)
    forward={int(a):int(b) for a,b in edges if outgoing[int(a)]==1 and incoming[int(b)]==1};children=set(forward.values());result=delta.copy()
    for start in sorted(set(forward)-children):
        path=[start]
        while path[-1] in forward:path.append(forward[path[-1]])
        if len(path)>=3:result[path]=.5*delta[path]+.5*median_filter(delta[path],size=(3,1),mode='nearest')
    return result

def apply_offsets(nodes,edges,delta,weight,temporal,shape):
    ids=sorted(nodes);index={k:i for i,k in enumerate(ids)};coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);pairs=np.asarray([(index[a],index[b]) for a,b in edges],int).reshape(-1,2)
    assert delta.shape==(len(ids),3) and np.isfinite(delta).all()
    d=temporal_offsets(coords,pairs,delta) if temporal else delta
    old=coords[:,1:];new=np.clip(np.rint(old+weight*d),0,np.asarray(shape[1:])-1).astype(int)
    for t in np.unique(coords[:,0]):
        at=np.flatnonzero(coords[:,0]==t);distance,nearest=cKDTree(old[at]*SCALE).query(new[at]*SCALE)
        own=np.linalg.norm((new[at]-old[at])*SCALE,axis=1);bad=distance+1e-8<own;new[at[bad]]=old[at[bad]]
        _,inverse,count=np.unique(new[at],axis=0,return_inverse=True,return_counts=True);bad=count[inverse]>1;new[at[bad]]=old[at[bad]]
    original_length=np.linalg.norm((old[pairs[:,1]]-old[pairs[:,0]])*SCALE,axis=1)
    for _ in range(32):
        length=np.linalg.norm((new[pairs[:,1]]-new[pairs[:,0]])*SCALE,axis=1);bad=(length>14)&(original_length<=14)
        if not bad.any():break
        reset=np.unique(pairs[bad]);new[reset]=old[reset]
    else:new=old.copy()
    result={k:dict(t=int(coords[i,0]),**dict(zip(('z','y','x'),map(int,new[i])))) for i,k in enumerate(ids)}
    shift=np.linalg.norm((new-old)*SCALE,axis=1)
    return result,dict(nodes=len(nodes),moved=int((shift>0).sum()),mean_shift_um=float(shift.mean()),max_shift_um=float(shift.max(initial=0)),topology_changed=False)

def measure(nodes,image):
    ids=sorted(nodes);coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);delta=np.zeros((len(ids),3))
    for t in np.unique(coords[:,0]):
        at=np.flatnonzero(coords[:,0]==t);delta[at]=image_offsets(np.asarray(image[int(t)]),coords[at,1:])
    return delta
