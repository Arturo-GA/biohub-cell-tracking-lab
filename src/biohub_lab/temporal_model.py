"""Image-conditioned parent attention and a symmetric mother/two-daughter head.

This is an independently implemented trainable model, not pretrained Trackastra/HOCT.
"""
import torch
from torch import nn
from .temporal_data import TemporalConfig


class TemporalLinker(nn.Module):
    def __init__(self,config=TemporalConfig()):
        super().__init__(); self.config=config; d=config.embedding
        layers=[]; previous=config.context
        for channels in (16,32,64):
            layers.extend([nn.Conv3d(previous,channels,3,stride=2,padding=1),
                nn.GroupNorm(4,channels),nn.SiLU()]); previous=channels
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(64,d))
        self.edge_projection=nn.Sequential(nn.Linear(4*d+7,d),nn.LayerNorm(d),nn.SiLU())
        self.orphan_projection=nn.Linear(d,d)
        layer=nn.TransformerEncoderLayer(d,4,dim_feedforward=2*d,dropout=.1,
            activation='gelu',batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.edge_output=nn.Linear(d,1)
        nn.init.zeros_(self.edge_output.weight); nn.init.zeros_(self.edge_output.bias)
        self.division=nn.Sequential(nn.Linear(4*d+4,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))

    def encode(self,patches):
        x=patches.float()/255.
        # Per-patch centering preserves temporal differences across the five channels.
        mean=x.mean(dim=(1,2,3,4),keepdim=True)
        std=x.std(dim=(1,2,3,4),keepdim=True).clamp_min(.05)
        return self.encoder((x-mean)/std)

    def parents(self,features,source,target,delta,mask):
        """source BxK and target B index the encoded node pool; delta is in microns."""
        a=features[source]; b=features[target][:,None,:].expand_as(a)
        distance=delta.square().sum(-1).sqrt()
        geom=torch.cat([delta/20.,delta.abs()/20.,distance[...,None]/20.],-1)
        tokens=self.edge_projection(torch.cat([a,b,a*b,(a-b).abs(),geom],-1))
        orphan=self.orphan_projection(features[target])[:,None,:]
        tokens=torch.cat([tokens,orphan],1)
        padding=torch.cat([~mask,torch.zeros((len(mask),1),device=mask.device,dtype=torch.bool)],1)
        residual=self.edge_output(self.attention(tokens,src_key_padding_mask=padding)).squeeze(-1)
        prior=-distance.square()/32.
        prior=torch.cat([prior,torch.full((len(prior),1),-5.,device=prior.device)],1)
        return (residual+prior).masked_fill(padding,-1.e4)

    def divisions(self,features,triples,delta):
        """Daughter exchange invariant; delta is Bx2x3 in physical coordinates."""
        p,a,b=features[triples[:,0]],features[triples[:,1]],features[triples[:,2]]
        distances=delta.square().sum(-1).sqrt()
        geom=torch.stack([distances.min(-1).values,distances.max(-1).values,
            (delta[:,0]-delta[:,1]).norm(dim=-1),delta.mean(1).norm(dim=-1)],-1)/20.
        return self.division(torch.cat([p,(a+b)/2,(a-b).abs(),a*b,geom],-1)).squeeze(-1)
