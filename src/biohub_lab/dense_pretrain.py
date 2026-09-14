"""Pretrain the existing temporal architecture on fully supervised synthetic movies."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import asdict

from .temporal_data import TemporalConfig
from .temporal_model import TemporalLinker
from .temporal_train import CacheStore


def pretrain(cache,output,steps=4000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(cache)/'cache_manifest.json').read_text())
    held=manifest['held_group']
    if any(x['video'].split('_')[0]==held for x in manifest['template_sources']):
        raise ValueError('Synthetic appearance leaks from held-out embryo')
    config=TemporalConfig();torch.manual_seed(config.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    # About 1.6 GB per fold; avoid thousands of simultaneously open mmap handles.
    model=TemporalLinker(config).to(device);store=CacheStore(cache,mmap_patches=False)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=2e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    rng=np.random.default_rng(7302026);positive=[];negative=[]
    for name in store.names:
        y=store.graphs[name]['triple_labels']
        positive.extend((name,int(i)) for i in np.flatnonzero(y==1))
        negative.extend((name,int(i)) for i in np.flatnonzero(y==0))
    if not positive or not negative: raise ValueError('Dense pretraining needs both classes')
    start=time.monotonic();history=[]
    for step in range(1,steps+1):
        name=str(rng.choice(store.names));g=store.graphs[name];targets=np.flatnonzero(g['labels']>=0)
        targets=rng.choice(targets,min(64,len(targets)),replace=False)
        refs=[positive[i] for i in rng.integers(len(positive),size=16)]+[negative[i] for i in rng.integers(len(negative),size=16)]
        batch=store.pack(name,targets,refs,device,drop_parent=True,rng=rng,augment=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            h=model.encode(batch['patches'])
            links=model.parents(h,batch['source'],batch['target'],batch['delta'],batch['mask'])
            divisions=model.divisions(h,batch['triples'],batch['triple_delta'])
            link_loss=F.cross_entropy(links.float(),batch['labels'])
            division_loss=F.binary_cross_entropy_with_logits(divisions.float(),batch['triple_labels'])
            loss=link_loss+.25*division_loss
        if not torch.isfinite(loss): raise ValueError('Nonfinite synthetic loss')
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt);scaler.update();scheduler.step()
        if step%100==0 or step==steps:
            row=dict(step=step,loss=float(loss.detach()),link_loss=float(link_loss.detach()),division_loss=float(division_loss.detach()),seconds=time.monotonic()-start)
            history.append(row);print('DENSE_PRETRAIN',held,json.dumps(row),flush=True)
    path=output/'pretrained.pt'
    torch.save(dict(state_dict=model.state_dict(),config=asdict(config),step=steps,held_group=held,
        template_sources=manifest['template_sources'],training_videos=manifest['training_videos'],
        generated_divisions=manifest['generated_divisions'],initialization='Dense procedural lineage pretraining from random initialization'),path)
    receipt=dict(steps=steps,seconds=time.monotonic()-start,scenes=len(store.names),
        positive_pairs=len(positive),negative_pairs=len(negative),generated_divisions=manifest['generated_divisions'],
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),held_group=held,
        selection='Fixed pretraining budget; no dev or held-out labels used to choose pretraining checkpoint')
    (output/'pretrain_history.json').write_text(json.dumps(history,indent=2))
    (output/'pretrain_receipt.json').write_text(json.dumps(receipt,indent=2))
    return path
