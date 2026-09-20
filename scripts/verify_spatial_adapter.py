"""Compare SDPA adapter against public inference blocks with explicit attention.

Imports only already-downloaded layer files. Stubs cover unused position helpers;
normalization, token handling and blocks use the published implementations.
"""
import importlib.util,os,sys,types
from pathlib import Path
import numpy as np
import torch
from biohub_lab.spatial_probe import SpatialBackbone

def main():
    os.environ['XFORMERS_DISABLED']='1';root=Path('artifacts/e038_source')
    for name in ['spatialdino','spatialdino.models','spatialdino.models.layers','spatialdino.data','spatialdino.utils']:
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    misc=types.ModuleType('spatialdino.utils.misc');misc.make_3tuple=lambda x:(x,x,x) if isinstance(x,int) else tuple(x);sys.modules[misc.__name__]=misc
    transforms=types.ModuleType('spatialdino.data.transforms')
    def unavailable(*a,**k):raise RuntimeError('Unexpected interpolation branch in no-position model')
    transforms.trilinear_interpolate_with_antialias=unavailable;sys.modules[transforms.__name__]=transforms
    swiglu=types.ModuleType('spatialdino.models.layers.swiglu_ffn');swiglu.SwiGLUFFNFused=unavailable;sys.modules[swiglu.__name__]=swiglu
    for name in ['attention','drop_path','layer_scale','mlp','block','patch_embed','pos_embed','encoder']:
        fullname='spatialdino.models.layers.'+name;p=root/('src__spatialdino__models__layers__'+name+'.py')
        spec=importlib.util.spec_from_file_location(fullname,p);mod=importlib.util.module_from_spec(spec);sys.modules[fullname]=mod;spec.loader.exec_module(mod)
    from spatialdino.models.layers.encoder import Encoder
    state=torch.load(root/'backbone.pth',weights_only=True,map_location='cpu');torch.set_num_threads(2);torch.manual_seed(38)
    official=Encoder(img_size=48,patch_size=8,stride=8,in_chans=1,embed_dim=384,depth=12,num_heads=6,init_values=1e-5,num_tt_register_tokens=1)
    missing=official.load_state_dict(state,strict=False);assert missing.missing_keys==['tt_register_tokens'] and not missing.unexpected_keys
    official.eval();adapter=SpatialBackbone(state);x=torch.rand(1,1,48,48,48)
    with torch.inference_mode():
        a=official._predict(x,vit_feat='patch',norm_feat='norm').flatten(2).transpose(1,2);b=adapter(x)
    error=float((a-b).abs().max());print('maximum_absolute_error',error);torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
if __name__=='__main__':main()
