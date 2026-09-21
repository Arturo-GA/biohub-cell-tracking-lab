"""Strict CPU weight compatibility check before allocating Kaggle GPU."""
import ast,hashlib,json,sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

repo=Path('outputs/e017_control_recovery/calibration_control/harmonic_control/tracking_repo')
sys.path.insert(0,str(repo/'src'))
from biohub_tracking.models import SimpleNodeTransformer,TemporalUNet3D
tree=ast.parse((repo/'scripts/train_unet_transformer.py').read_text(encoding='utf8'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UNetNodeTransformer')
namespace=dict(torch=torch,nn=nn,F=F,SimpleNodeTransformer=SimpleNodeTransformer)
exec(compile(ast.Module(body=[node],type_ignores=[]),'reviewed_model_class','exec'),namespace)
model=namespace['UNetNodeTransformer'](TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),32,32)
path=Path('artifacts/synthetic_edge/synthetic_5fold_swa.pth')
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest=='0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9'
state=torch.load(path,map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True)
assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
result=dict(checkpoint_sha256=digest,strict_load=True,tensors_loaded=len(state),parameters=sum(p.numel() for p in model.parameters()),default_config=dict(unet_out_channels=32,unet_layers=[32,64,128],downsample=[1,4,4],window_size=2),training_membership='Public synthetic fine tuning; base real-image training membership unknown')
Path('results/E056_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
