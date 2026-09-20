"""Matched native-resolution regressors and coordinate-aligned coarse control."""
import torch
from torch import nn
from torch.nn import functional as F

def coarse_control(x):
    low=x[:,:,:,::4,::4]
    # Explicit source coordinates: native index k maps to coarse index k/4.
    z=torch.linspace(-1,1,16,device=x.device);a=torch.arange(64,device=x.device)/4/15*2-1
    zz,yy,xx=torch.meshgrid(z,a,a,indexing='ij');grid=torch.stack([xx,yy,zz],-1)[None].expand(len(x),-1,-1,-1,-1)
    return F.grid_sample(low,grid,align_corners=True,padding_mode='border')

class ResolutionProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(2,16,3,stride=(1,2,2),padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv3d(16,32,3,stride=(1,2,2),padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv3d(32,48,3,stride=2,padding=1),nn.GroupNorm(4,48),nn.SiLU(),nn.AdaptiveAvgPool3d(4),nn.Flatten(),nn.Linear(48*64,128),nn.SiLU(),nn.Linear(128,3))
    def forward(self,x):return 4.5*torch.tanh(self.net(x))
