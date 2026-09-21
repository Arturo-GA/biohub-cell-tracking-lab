"""Real-image adaptation of the externally supervised dense center detector."""
import numpy as np
import torch
from torch import nn
from biohub_lab.dense_center import DenseCenter

class TemporalDenseCenter(DenseCenter):
    def __init__(self):
        super().__init__()
        self.first[0]=nn.Conv3d(3,16,3,padding=1)

    def initialize(self,state):
        expanded={k:v.clone() for k,v in state.items()}
        weight=expanded['first.0.weight']
        expanded['first.0.weight']=torch.cat([torch.zeros_like(weight),weight,torch.zeros_like(weight)],1)
        self.load_state_dict(expanded,strict=True)

def crop_three(volume,t,origin,size=32):
    indices=np.clip([t-1,t,t+1],0,len(volume)-1)
    low=np.asarray(origin,int);high=low+size;shape=np.asarray(volume.shape[1:])
    start=np.maximum(low,0);stop=np.minimum(high,shape)
    assert np.all(stop>start)
    value=np.array(volume[(indices,*[slice(int(a),int(b)) for a,b in zip(start,stop)])],np.float32)
    pad=[(0,0)]+[(int(a),int(b)) for a,b in zip(start-low,high-stop)]
    result=np.pad(value,pad,mode='edge');assert result.shape==(3,size,size,size)
    return result

def positive_heatmap(centers,size=32):
    """Positive neighborhoods only; zero values elsewhere are NOT negative labels."""
    y=np.zeros((size,size,size),np.float32)
    for center in centers:
        low=np.maximum(np.floor(center-3).astype(int),0);high=np.minimum(np.ceil(center+3).astype(int)+1,size)
        if np.any(high<=low):continue
        grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing='ij'))
        value=np.exp(-np.sum((grid-center.reshape(3,1,1,1))**2,axis=0)/2)
        sl=tuple(slice(a,b) for a,b in zip(low,high));y[sl]=np.maximum(y[sl],value)
    return y
