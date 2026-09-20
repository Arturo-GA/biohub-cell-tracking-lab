"""Sparse-supervised 3D center vector field: no unlabeled-background targets."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CenterField(nn.Module):
    def __init__(self,channels):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv3d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv3d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.first=block(channels,16);self.down=block(16,32);self.bottom=block(32,64);self.up=block(96,32);self.last=block(48,16);self.head=nn.Conv3d(16,3,1)
    def forward(self,x):
        a=self.first(x);b=self.down(F.avg_pool3d(a,2));c=self.bottom(F.avg_pool3d(b,2))
        d=self.up(torch.cat([F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],1))
        h=self.last(torch.cat([F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a],1))
        return 4*torch.tanh(self.head(h))

def vector_targets(centers,shape=(32,32,32),radius=3.):
    """Supervise only a small neighborhood of each known center, masking all else."""
    target=np.zeros((3,*shape),np.float32);distance=np.full(shape,np.inf,np.float32)
    for center in centers:
        lo=np.maximum(np.floor(center-radius).astype(int),0);hi=np.minimum(np.ceil(center+radius).astype(int)+1,shape)
        if np.any(hi<=lo):continue
        grid=np.stack(np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)),indexing='ij'))
        delta=center.reshape(3,1,1,1)-grid;dist=np.linalg.norm(delta,axis=0);s=tuple(slice(a,b) for a,b in zip(lo,hi))
        keep=(dist<=radius)&(dist<distance[s]);target[(slice(None),*s)][:,keep]=delta[:,keep];distance[s][keep]=dist[keep]
    return target,np.isfinite(distance)

def votes(field,image):
    from scipy.ndimage import gaussian_filter,maximum_filter
    grid=np.indices(image.shape,dtype=np.float32);end=grid+field
    # Trilinear splatting avoids rounding artifacts. Weight uses only the image.
    base=np.floor(end).astype(int);frac=end-base;score=np.zeros(image.shape,np.float32)
    weight=np.maximum(image-gaussian_filter(image,3),0)
    for dz in [0,1]:
        for dy in [0,1]:
            for dx in [0,1]:
                shift=np.array([dz,dy,dx]).reshape(3,1,1,1);at=base+shift
                good=np.all((at>=0)&(at<np.asarray(image.shape).reshape(3,1,1,1)),axis=0)
                w=np.prod(np.where(shift,frac,1-frac),axis=0)*weight
                np.add.at(score,tuple(at[:,good]),w[good])
    score=gaussian_filter(score,.6);xyz=np.argwhere((score==maximum_filter(score,3))&(score>0));order=np.argsort(-score[tuple(xyz.T)],kind='stable');return xyz[order[:512]],score[tuple(xyz[order[:512]].T)]
