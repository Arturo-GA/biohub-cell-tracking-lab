import ast,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from biohub_lab.public_divnet import input_patch,PublicDivNet
source=Path('artifacts/research_latest_20260920/canhtoanle__biohub-div-complete-v33a/source.py');tree=ast.parse(source.read_text(encoding='utf8'))
functions={'_marker_3d','_norm_crop','_frame_stat','_reflect_index','_padded_crop','build_input_batch'}
namespace=dict(np=np,torch=torch,nn=nn,CROP_Z=16,CROP_Y=32,CROP_X=32,IMAGE_LAGS=(-1,0,1,2),MARKER_SIGMA_ZYX=(1.5,2.,2.),LO_PCT=50.,HI_PCT=99.5,CLIP_LO=-.5,CLIP_HI=6.,NORM_EPS=1e-6,BASE_CHANNELS=16)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions];assert len(nodes)==6;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
rng=np.random.default_rng(48);vol=rng.random((4,20,40,40),dtype=np.float32);coords=np.array([[0,0,1,2],[3,19,156,156],[2,8.5,77,85]])
errors={}
for mode,refmode in [('frame','frame'),('crop','per_frame')]:
    ours=np.stack([input_patch(vol,c,mode,{}) for c in coords]);ref=namespace['build_input_batch'](vol,coords[:,1:]/[1,4,4],coords[:,0],norm=refmode)
    np.testing.assert_allclose(ours,ref,atol=2e-6);errors[mode]=float(np.max(abs(ours-ref)))
Path('results/public_divnet_preprocess_audit.json').write_text(json.dumps(dict(reference=str(source),scope='Same supplied XY4 volume; does not prove original training preprocessing used XY stride rather than average pooling.',max_absolute_input_difference=errors),indent=2)+'\n');print(errors)
