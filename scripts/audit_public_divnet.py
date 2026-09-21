"""Read-only audit: execute only the two reviewed model class definitions."""
import ast,hashlib,json
from pathlib import Path
import torch
root=Path('artifacts/research_latest_20260920');source=root/'andnyu__biohub-density-adaptive-0-948-reproduction/source.py';tree=ast.parse(source.read_text(encoding='utf8'))
classes=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name in ['_DivNetConvBlock','DivNetMitosisClassifier']]
assert len(classes)==2;namespace={'torch':torch};exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),namespace)
model=namespace['DivNetMitosisClassifier']();before={k:v.clone() for k,v in model.state_dict().items()};weight=root/'divnet/best_overall.pt';saved=torch.load(weight,map_location='cpu',weights_only=True)
report=model.load_state_dict(saved['model_state'],strict=False);shared=set(model.state_dict())&set(saved['model_state'])
changed=[k for k,v in model.state_dict().items() if not torch.equal(before[k],v)]
audit=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),model_keys=len(model.state_dict()),checkpoint_keys=len(saved['model_state']),matching_keys=sorted(shared),changed_model_tensors=changed,missing_keys=report.missing_keys,unexpected_keys=report.unexpected_keys,declared_input_channels=saved['config']['input']['channels'],notebook_input_channels=1,conclusion='No checkpoint tensor loaded into notebook classifier; strict=False silently accepts incompatible names. A success print is not a trained model.')
Path('results/public_divnet_compatibility_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['missing_keys','unexpected_keys']},indent=2))
