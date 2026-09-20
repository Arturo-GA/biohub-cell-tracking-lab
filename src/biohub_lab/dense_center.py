"""Dense center logits with the same convolutional backbone as E039."""
from torch import nn
from torch.nn import functional as F
import torch
from biohub_lab.temporal_detector import CenterField

class DenseCenter(CenterField):
    def __init__(self):
        super().__init__(1);self.head=nn.Conv3d(16,1,1)
    def forward(self,x):
        a=self.first(x);b=self.down(F.avg_pool3d(a,2));c=self.bottom(F.avg_pool3d(b,2))
        d=self.up(torch.cat([F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],1))
        h=self.last(torch.cat([F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a],1))
        return self.head(h)[:,0]
