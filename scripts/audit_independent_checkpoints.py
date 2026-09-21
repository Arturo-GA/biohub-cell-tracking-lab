"""Reproduce strict model loading and a small CPU forward-pass sanity check."""
import json
from pathlib import Path
import numpy as np
import torch
from biohub_lab.independent_expert import load,probabilities,sparse

repo=Path('outputs/e017_control_recovery/calibration_control/harmonic_control/tracking_repo')
cfg=json.loads(Path('baseline/e059_protocol.json').read_text())
torch.set_num_threads(2)
rows=[]
for spec in cfg['models'].values():
    model,pos=load(repo,Path('artifacts/synthetic_edge')/spec['file'],spec['sha256'],'cpu')
    with torch.inference_mode():
        features,_=model.encode(torch.zeros(1,2,16,16,16))
        p=probabilities(model,pos,features,np.array([[0,3,16,16],[0,6,32,32]],np.float32),
            np.array([[1,4,16,16],[1,7,32,32]],np.float32))
    assert np.isfinite(p).all() and np.allclose(p.sum(0),1)
    edges,_=sparse(p,[[0,1]],1)
    assert (edges==[0,1]).all(1).any()
    rows.append(dict(file=spec['file'],sha256=spec['sha256'],parameters=sum(x.numel() for x in model.parameters()),smoke='passed'))
Path('results/E059_checkpoint_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
