"""Checkpoint-compatible public DivNet architecture; see NOTICE and E048 protocol."""
import numpy as np
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();layers=[]
        for cin in [a,b]:layers.extend([nn.Conv3d(cin,b,3,padding=1,bias=False),nn.InstanceNorm3d(b,affine=True),nn.ReLU(inplace=True)])
        self.block=nn.Sequential(*layers)
    def forward(self,x):return self.block(x)

class PublicDivNet(nn.Module):
    def __init__(self):
        super().__init__()
        for name,a,b in [('enc1',5,16),('enc2',16,32),('enc3',32,64),('bottleneck',64,128),('dec3',128,64),('dec2',64,32),('dec1',32,16)]:setattr(self,name,Block(a,b))
        for i,a,b in [(3,128,64),(2,64,32),(1,32,16)]:setattr(self,'up'+str(i),nn.ConvTranspose3d(a,b,2,stride=2))
        self.pool=nn.MaxPool3d(2);self.head=nn.Linear(16,1)
    def forward(self,x):
        skip=[]
        for name in ['enc1','enc2','enc3']:x=getattr(self,name)(x);skip.append(x);x=self.pool(x)
        x=self.bottleneck(x)
        for i,prior in zip([3,2,1],reversed(skip)):x=getattr(self,'dec'+str(i))(torch.cat([getattr(self,'up'+str(i))(x),prior],1))
        return self.head(x.mean((2,3,4)))[:,0]

def input_patch(volume,coord,mode='frame',cache=None):
    """XY4 input, four lags plus fractional center marker; no labels."""
    center=np.asarray(coord[1:])/np.array([1,4,4]);shape=np.array([16,32,32]);origin=np.rint(center).astype(int)-shape//2
    indices=[]
    for start,n,size in zip(origin,shape,volume.shape[1:]):
        raw=np.arange(start,start+n);period=2*(size-1)
        index=np.zeros(n,int) if size==1 else np.abs((raw+size-1)%period-(size-1))
        indices.append(index)
    patches=[]
    for dt in [-1,0,1,2]:
        t=int(np.clip(coord[0]+dt,0,len(volume)-1));frame=np.asarray(volume[t],np.float32);patch=frame[np.ix_(*indices)]
        if mode=='frame':
            if cache is not None and t in cache:lo,hi=cache[t]
            else:
                lo,hi=np.percentile(frame,[50,99.5])
                if cache is not None:cache[t]=(lo,hi)
        else:lo,hi=np.percentile(patch,[50,99.5])
        patches.append(np.clip((patch-lo)/max(hi-lo,1e-6),-.5,6))
    grid=np.indices(tuple(shape),dtype=np.float32);delta=(grid-(center-origin).reshape(3,1,1,1))/np.array([1.5,2,2]).reshape(3,1,1,1)
    patches.append(np.exp(-.5*np.sum(delta**2,axis=0)))
    return np.asarray(patches,np.float32)
