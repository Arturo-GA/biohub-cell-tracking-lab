"""Six-timepoint, three-view event verifier with a frozen E006 representation."""
import torch
from torch import nn
from .temporal_data import TemporalConfig
from .temporal_model import TemporalLinker


class MitosisSpecialist(nn.Module):
    def __init__(self,config=TemporalConfig()):
        super().__init__()
        if config.context!=5: raise ValueError('The event alignment requires five-frame node patches')
        self.config=config
        self.backbone=TemporalLinker(config)
        self.backbone.requires_grad_(False)
        layers=[]; previous=1
        for channels in (8,16,32):
            layers.extend([nn.Conv3d(previous,channels,3,stride=2,padding=1),
                           nn.GroupNorm(4,channels),nn.SiLU()])
            previous=channels
        self.frame_encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool3d(1),nn.Flatten())
        self.time_encoder=nn.GRU(4*32+2,64,batch_first=True,bidirectional=True)
        self.event_head=nn.Sequential(nn.Linear(128+4*config.embedding+5,192),
            nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.2),nn.Linear(192,1))
        self.backbone.eval()

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self,patches,delta):
        """B x (mother, daughter A, daughter B) x 5 x ZYX; delta B x 2 x 3.

        Mother windows cover t-2..t+2; daughter windows cover t-1..t+3.
        Align them on six absolute times and explicitly mark absent boundary views.
        Daughter exchange is invariant, while temporal order is represented by a GRU.
        """
        if patches.ndim!=6 or patches.shape[1:3]!=(3,5):
            raise ValueError('Expected B x 3 x 5 x Z x Y x X patches')
        if delta.shape!=(len(patches),2,3): raise ValueError('Invalid daughter displacement shape')
        batch=len(patches)
        flat=patches.reshape(batch*3,5,*patches.shape[-3:])
        with torch.no_grad():
            h=self.backbone.encode(flat)
            refs=torch.arange(batch*3,device=patches.device).reshape(batch,3)
            old_score=self.backbone.divisions(h,refs,delta)
            p,a,b=h.reshape(batch,3,-1).unbind(1)
            frozen=torch.cat([p,(a+b)/2,(a-b).abs(),a*b],-1)
        x=patches.float()/255.
        mean=x.mean(dim=(1,2,3,4,5),keepdim=True)
        std=x.std(dim=(1,2,3,4,5),keepdim=True).clamp_min(.05)
        x=(x-mean)/std
        frames=self.frame_encoder(x.reshape(batch*15,1,*x.shape[-3:])).reshape(batch,3,5,32)
        zero=frames.new_zeros(batch,1,32)
        mother=torch.cat([frames[:,0],zero],1)
        da=torch.cat([zero,frames[:,1]],1)
        db=torch.cat([zero,frames[:,2]],1)
        present=frames.new_tensor([[1,0],[1,1],[1,1],[1,1],[1,1],[0,1]])[None].expand(batch,-1,-1)
        sequence=torch.cat([mother,(da+db)/2,(da-db).abs(),da*db,present],-1)
        # Run the recurrent kernel in FP32 across CPU/CUDA/cuDNN versions.
        with torch.autocast(device_type=sequence.device.type,enabled=False):
            _,state=self.time_encoder(sequence.float())
        temporal=torch.cat([state[0],state[1]],-1)
        dist=delta.norm(dim=-1)
        geom=torch.stack([dist.min(-1).values,dist.max(-1).values,
            (delta[:,0]-delta[:,1]).norm(dim=-1),delta.mean(1).norm(dim=-1)],-1)/20.
        return self.event_head(torch.cat([temporal,frozen,geom,old_score[:,None]],-1)).squeeze(-1)
