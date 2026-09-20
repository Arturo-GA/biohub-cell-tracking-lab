"""Inference-only SpatialDINO ViT-S/8 adapter, using native PyTorch SDPA.

Architecture verified against kirchhausenlab/spatialdino commit
ca3ab86b34430d963f12a3909baaeb9343c63b7d. No external weights embedded.
The inference zero register follows that repository's inference.yaml.
"""
import torch
from torch.nn import functional as F

class SpatialBackbone:
    def __init__(self,state,device='cpu'):
        expected={'mask_token','cls_token','patch_embed.proj.weight','patch_embed.proj.bias','norm.weight','norm.bias'}
        for i in range(12):
            for name in ['norm1','norm2','attn.qkv','attn.proj','mlp.fc1','mlp.fc2']:
                expected.update(f'blocks.{i}.{name}.{suffix}' for suffix in ['weight','bias'])
            expected.update([f'blocks.{i}.ls1.gamma',f'blocks.{i}.ls2.gamma'])
        assert set(state)==expected, (set(state)-expected,expected-set(state))
        self.s={k:v.to(device).float() for k,v in state.items()}
    def __call__(self,volume):
        s=self.s
        def linear(x,p):return F.linear(x,s[p+'.weight'],s[p+'.bias'])
        def norm(x,p):return F.layer_norm(x,(384,),s[p+'.weight'],s[p+'.bias'],1e-6)
        x=F.conv3d(volume,s['patch_embed.proj.weight'],s['patch_embed.proj.bias'],stride=8).flatten(2).transpose(1,2)
        x=torch.cat([s['cls_token'].expand(len(x),-1,-1),torch.zeros_like(s['cls_token']).expand(len(x),-1,-1),x],1)
        for i in range(12):
            p=f'blocks.{i}'; h=norm(x,p+'.norm1');b,n,c=h.shape
            q,k,v=linear(h,p+'.attn.qkv').reshape(b,n,3,6,64).permute(2,0,3,1,4).unbind(0)
            h=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,c)
            x=x+linear(h,p+'.attn.proj')*s[p+'.ls1.gamma']
            x=x+linear(F.gelu(linear(norm(x,p+'.norm2'),p+'.mlp.fc1')),p+'.mlp.fc2')*s[p+'.ls2.gamma']
        return norm(x,'norm')[:,2:]

def summarize(tokens):
    # Preserve spatial order for localization instead of collapsing it to a CLS token.
    b,n,c=tokens.shape; side=round(n**(1/3));assert side**3==n
    return F.adaptive_avg_pool3d(tokens.transpose(1,2).reshape(b,c,side,side,side),2).flatten(1)
