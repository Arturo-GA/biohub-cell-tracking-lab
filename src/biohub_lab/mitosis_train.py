"""Train only an event specialist; choose checkpoint and threshold inside development."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .mitosis_model import MitosisSpecialist
from .temporal_data import SCALE,TemporalConfig
from .temporal_train import CacheStore,make_split

E006_HASHES={
    '44b6':'59b4e50e6fe27966b73e4531ccb070b01a589b84e04b81cd1a7da7beafe99fd5',
    '6bba':'835d290edadc76581a5653482c94b14144991880ac990b4bd756da14762f2cbd',
}


def ranking_metrics(labels,scores):
    """Tie-aware AP and F0.5 operating point on known labels, not calibrated probabilities."""
    y=np.asarray(labels,dtype=np.int64); s=np.asarray(scores,dtype=np.float64)
    if y.shape!=s.shape or y.ndim!=1 or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():
        raise ValueError('Invalid labelled scores')
    pos=int(y.sum()); neg=len(y)-pos
    if not pos or not neg:
        return dict(positive=pos,negative=neg,average_precision=None,balanced_bce=None,
            threshold=None,threshold_f05=None,threshold_tp=0,threshold_fp=0,threshold_fn=pos)
    order=np.argsort(-s,kind='stable'); sy=y[order]; ss=s[order]
    end=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(sy)[end]; fp=end+1-tp; fn=pos-tp
    precision=tp/(end+1)
    ap=float(np.sum(np.diff(np.r_[0,tp])/pos*precision))
    f05=1.25*tp/(1.25*tp+.25*fn+fp)
    # First maximum has the highest threshold; exact score ties remain indivisible.
    selected=int(np.argmax(f05)); loss=np.logaddexp(0,s)-s*y
    return dict(positive=pos,negative=neg,average_precision=ap,
        balanced_bce=float(.5*(loss[y==1].mean()+loss[y==0].mean())),
        threshold=float(ss[end[selected]]),threshold_f05=float(f05[selected]),
        threshold_tp=int(tp[selected]),threshold_fp=int(fp[selected]),threshold_fn=int(fn[selected]))


def operating_counts(labels,scores,threshold):
    y=np.asarray(labels)==1; predicted=(np.asarray(scores)>=threshold if threshold is not None else np.zeros(len(y),bool))
    tp=int((y&predicted).sum()); fp=int((~y&predicted).sum()); fn=int((y&~predicted).sum())
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


def references(store,names):
    positive=[]; negative=[]
    for name in names:
        labels=store.graphs[name]['triple_labels']
        positive.extend((name,int(i)) for i in np.flatnonzero(labels==1))
        negative.extend((name,int(i)) for i in np.flatnonzero(labels==0))
    if not positive or not negative: raise ValueError('Specialist training requires both known classes')
    return positive,negative


def sample_refs(pools,rng,per_class):
    return [pool[i] for pool in pools for i in rng.integers(len(pool),size=per_class)]


def pack_events(store,refs,device,rng=None):
    patches=[]; delta=[]; labels=[]
    for name,row in refs:
        graph=store.graphs[name]; mother,a,b=graph['triples'][row]
        patches.append(store.patches[name][[mother,a,b]])
        delta.append((graph['coords'][[a,b],1:]-graph['coords'][mother,1:])*SCALE)
        labels.append(graph['triple_labels'][row])
    patches=np.stack(patches); delta=np.asarray(delta,np.float32)
    if rng is not None:
        for axis in range(3):
            if rng.random()<.5:
                patches=np.flip(patches,axis=axis+3);delta[:,:,axis]*=-1
        # One consistent intensity transform preserves temporal/relative-view evidence.
        gain=rng.uniform(.8,1.2,size=(len(refs),1,1,1,1,1))
        patches=np.rint(np.clip(patches*gain,0,255)).astype(np.uint8)
    return (torch.as_tensor(np.ascontiguousarray(patches),device=device),
            torch.as_tensor(np.ascontiguousarray(delta),device=device),
            torch.as_tensor(labels,dtype=torch.float32,device=device))


@torch.inference_mode()
def evaluate_events(model,store,names,device):
    model.eval(); scores=[];labels=[];video=[];rows=[]
    for name in names:
        count=len(store.graphs[name]['triples'])
        for start in range(0,count,48):
            refs=[(name,i) for i in range(start,min(start+48,count))]
            patch,delta,y=pack_events(store,refs,device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                score=model(patch,delta)
            scores.extend(score.float().cpu().tolist());labels.extend(y.cpu().tolist())
            video.extend([name]*len(refs));rows.extend(i for _,i in refs)
    arrays=dict(scores=np.asarray(scores,np.float32),labels=np.asarray(labels,np.int64),
                video=np.asarray(video),row=np.asarray(rows,np.int64))
    return ranking_metrics(arrays['labels'],arrays['scores']),arrays


def train_specialist(real_root,synthetic_root,initial_path,output,held_group,
                     pretrain_steps=4000,finetune_steps=3000,per_class=16,replay_per_class=8,
                     expected_hash=None,eval_every=500):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    actual=hashlib.sha256(Path(initial_path).read_bytes()).hexdigest()
    expected=E006_HASHES.get(held_group) if expected_hash is None else expected_hash
    if actual!=expected: raise ValueError('E006 checkpoint checksum mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    config=TemporalConfig();torch.manual_seed(config.seed)
    real=CacheStore(real_root); split=make_split(real.names,held_group)
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    if initial['split']!=split or TemporalConfig(**initial['config'])!=config:
        raise ValueError('Frozen E006 checkpoint split/config mismatch')
    manifest=json.loads((Path(synthetic_root)/'cache_manifest.json').read_text())
    if (manifest['held_group']!=held_group or set(manifest['training_videos'])!=set(split['train']) or
        any(t['video'] not in split['train'] for t in manifest['template_sources'])):
        raise ValueError('Synthetic appearance outside specialist training split')
    synthetic=CacheStore(synthetic_root,mmap_patches=False)
    model=MitosisSpecialist(config).to(device)
    model.backbone.load_state_dict(initial['state_dict'],strict=True)
    pools_real=references(real,split['train']);pools_synthetic=references(synthetic,synthetic.names)
    rng=np.random.default_rng(config.seed+7);history=[];start=time.monotonic()
    selected=None;best_key=None;threshold=None
    for stage,steps in (('synthetic',pretrain_steps),('real',finetune_steps)):
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4 if stage=='synthetic' else 1e-4,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps,eta_min=1e-5)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        for step in range(1,steps+1):
            model.train()
            if stage=='synthetic':
                patch,delta,y=pack_events(synthetic,sample_refs(pools_synthetic,rng,per_class),device,rng)
            else:
                batches=[pack_events(real,sample_refs(pools_real,rng,per_class),device,rng),
                         pack_events(synthetic,sample_refs(pools_synthetic,rng,replay_per_class),device,rng)]
                patch,delta,y=(torch.cat([b[k] for b in batches]) for k in range(3))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                score=model(patch,delta).float()
                bce=F.binary_cross_entropy_with_logits(score,y)
                ranking=F.softplus(score[y==0][:,None]-score[y==1][None]).mean()
                loss=bce+.25*ranking
            if not torch.isfinite(loss): raise ValueError('Nonfinite specialist training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            scaler.step(optimizer);scaler.update();scheduler.step()
            if step%100==0 or step==steps:
                row=dict(stage=stage,step=step,loss=float(loss.detach()),bce=float(bce.detach()),ranking=float(ranking.detach()),seconds=time.monotonic()-start)
                history.append(row);print('SPECIALIST_TRAIN',held_group,json.dumps(row),flush=True)
            if stage=='real' and (step%eval_every==0 or step==steps):
                dev,arrays=evaluate_events(model,real,split['dev'],device)
                if dev['average_precision'] is None: raise ValueError('Development lacks known positives or negatives')
                key=(dev['average_precision'],-dev['balanced_bce'])
                print('SPECIALIST_DEV',held_group,step,json.dumps(dev),flush=True)
                if best_key is None or key>best_key:
                    best_key=key;selected=step;threshold=dev['threshold']
                    torch.save(dict(state_dict=model.state_dict(),config=asdict(config),split=split,
                        step=step,pretrain_steps=pretrain_steps,finetune_steps=finetune_steps,
                        initial_checkpoint_sha256=actual,dev=dev,threshold=threshold,
                        architecture='six-timepoint three-view GRU plus frozen E006 representation'),output/'best.pt')
                    np.savez_compressed(output/'dev_scores.npz',**arrays)
        if stage=='synthetic':
            torch.save(dict(state_dict=model.state_dict(),config=asdict(config),step=steps,
                split=split,initial_checkpoint_sha256=actual),output/'pretrained.pt')
    checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    for name,tensor in model.backbone.state_dict().items():
        if not torch.equal(tensor.cpu(),initial['state_dict'][name].cpu()):
            raise ValueError('Frozen E006 representation changed during specialist training')
    holdout,arrays=evaluate_events(model,real,split['holdout'],device)
    # Heldout threshold suggestions must never be presented as an operating decision.
    holdout={k:v for k,v in holdout.items() if not k.startswith('threshold')}
    holdout['at_dev_threshold']=operating_counts(arrays['labels'],arrays['scores'],threshold)
    np.savez_compressed(output/'holdout_scores.npz',**arrays)
    receipt=dict(experiment='E007',held_group=held_group,split=split,initial_checkpoint_sha256=actual,
        frozen_backbone_unchanged=True,pretrain_steps=pretrain_steps,finetune_steps=finetune_steps,
        real_per_class=per_class,synthetic_replay_per_class=replay_per_class,
        real_training_positive=len(pools_real[0]),real_training_negative=len(pools_real[1]),
        synthetic_training_positive=len(pools_synthetic[0]),synthetic_training_negative=len(pools_synthetic[1]),
        selected_step=selected,dev=checkpoint['dev'],threshold=threshold,holdout=holdout,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seconds=time.monotonic()-start,device=str(device),checkpoint_sha256=hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
        selection='Max development AP, then lower balanced BCE; F0.5 threshold on that same development set only.',
        scope='Whole-embryo holdout on known triple labels; operating precision is not calibrated to unknown detector candidates.')
    (output/'training_receipt.json').write_text(json.dumps(receipt,indent=2))
    (output/'history.json').write_text(json.dumps(history,indent=2))
    print('SPECIALIST_FOLD_COMPLETE',held_group,json.dumps(holdout),flush=True)
    return receipt
