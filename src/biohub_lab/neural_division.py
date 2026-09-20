"""Symmetric neural event representations; no labels or graph edits here."""
import numpy as np

def event_features(h,contexts,sequence):
    h=np.asarray(h,dtype=np.float32);contexts=np.asarray(contexts,dtype=np.int64).reshape(-1,9)
    width=h.shape[1]
    if not len(contexts):return np.empty((0,width*(9 if sequence else 3)),np.float32)
    f=h[contexts]
    if sequence:
        return np.concatenate([f[:,:3].reshape(len(f),-1),((f[:,3:6]+f[:,6:9])/2).reshape(len(f),-1),np.abs(f[:,3:6]-f[:,6:9]).reshape(len(f),-1)],axis=1)
    return np.concatenate([f[:,2],(f[:,3]+f[:,6])/2,np.abs(f[:,3]-f[:,6])],axis=1)

def development_threshold(y,scores):
    y=np.asarray(y);scores=np.asarray(scores);options=[]
    for value in np.unique(scores):
        selected=scores>=value;tp=int(((y==1)&selected).sum());fp=int(((y==0)&selected).sum())
        if tp>=2 and fp==0:options.append((tp,float(value)))
    return sorted(options,key=lambda v:(-v[0],-v[1]))[0][1] if options else None
