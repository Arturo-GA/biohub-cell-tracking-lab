"""Train from scratch with a held-out acquisition prefix and an internal dev split."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .temporal_data import SCALE, TemporalConfig
from .temporal_model import TemporalLinker


class CacheStore:
    def __init__(self,root,names=None,mmap_patches=True):
        self.root=Path(root)
        manifest=json.loads((self.root/'cache_manifest.json').read_text())
        if not manifest['complete']: raise ValueError('Incomplete cache')
        self.names=sorted(names or [r['name'] for r in manifest['videos']])
        self.graphs={}; self.patches={}
        for name in self.names:
            with np.load(self.root/name/'graph.npz') as z: self.graphs[name]={k:z[k] for k in z.files}
            self.patches[name]=np.load(self.root/name/'patches.npy',mmap_mode='r' if mmap_patches else None)

    def pack(self,name,targets,triple_refs,device,drop_parent=False,rng=None,augment=False):
        g=self.graphs[name]; targets=np.asarray(targets)
        parent=g['parents'][targets].copy(); mask=parent>=0
        labels=g['labels'][targets].copy()
        if drop_parent:
            take=(rng.random(len(targets))<.1)&(labels>=0)
            mask[np.flatnonzero(take),labels[take]]=False
            labels[take]=parent.shape[1]
        refs=set((name,int(n)) for n in parent[mask]); refs.update((name,int(t)) for t in targets)
        for video,row in triple_refs:
            refs.update((video,int(n)) for n in self.graphs[video]['triples'][row])
        refs=sorted(refs); lookup={r:i for i,r in enumerate(refs)}
        patch=np.stack([self.patches[v][n] for v,n in refs])
        source=np.zeros(parent.shape,np.int64)
        for i,j in np.argwhere(mask): source[i,j]=lookup[(name,int(parent[i,j]))]
        dest=np.array([lookup[(name,int(t))] for t in targets],np.int64)
        safe=parent.clip(0)
        delta=(g['coords'][safe,1:]-g['coords'][targets,1:][:,None])*SCALE
        triples=[]; triple_delta=[]; triple_labels=[]
        for v,row in triple_refs:
            gg=self.graphs[v]; p,a,b=gg['triples'][row]
            triples.append([lookup[(v,int(n))] for n in (p,a,b)])
            triple_delta.append((gg['coords'][[a,b],1:]-gg['coords'][p,1:])*SCALE)
            triple_labels.append(gg['triple_labels'][row])
        triple_delta=np.asarray(triple_delta,np.float32).reshape(-1,2,3)
        if augment:
            for axis in range(3):
                if rng.random()<.5:
                    patch=np.flip(patch,axis=axis+2); delta[:,:,axis]*=-1; triple_delta[:,:,axis]*=-1
        def tensor(x,dtype=None): return torch.as_tensor(np.ascontiguousarray(x),dtype=dtype,device=device)
        return dict(patches=tensor(patch),source=tensor(source),target=tensor(dest),delta=tensor(delta),
            mask=tensor(mask),labels=tensor(labels),triples=tensor(np.asarray(triples,np.int64).reshape(-1,3)),
            triple_delta=tensor(triple_delta),triple_labels=tensor(triple_labels,torch.float32))


def make_split(names,held_group):
    holdout=sorted(n for n in names if n.split('_')[0]==held_group)
    available=sorted(n for n in names if n not in holdout)
    if not holdout or len(available)<5: raise ValueError('Insufficient acquisition groups')
    ordered=sorted(available,key=lambda n:hashlib.sha256(('temporal-dev-2026:'+n).encode()).hexdigest())
    dev=sorted(ordered[:max(2,len(ordered)//5)]); train=sorted(set(available)-set(dev))
    if {n.split('_')[0] for n in train+dev}&{held_group}: raise ValueError('Acquisition leakage')
    return dict(train=train,dev=dev,holdout=holdout,held_group=held_group,
        grouping='Whole embryo prefix; two training embryos confirmed by competition host',
        grouping_source='https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793',
        scope='Model and raw-image encoder trained from scratch; holdout labels excluded from optimization and checkpoint selection')


def summarize_predictions(records):
    total=sum(r['n'] for r in records); covered=sum(r['covered'] for r in records)
    return dict(n=total,covered=covered,coverage=covered/max(total,1),
        parent_accuracy=sum(r['correct'] for r in records)/max(total,1),
        nearest_accuracy=sum(r['nearest_correct'] for r in records)/max(total,1),
        ce=sum(r['ce_sum'] for r in records)/max(covered,1))


@torch.inference_mode()
def validate(model,store,names,device,config,full=True):
    model.eval(); rows=[]
    for name in names:
        g=store.graphs[name]; targets=g['targets']; valid=targets[g['labels'][targets]>=0]
        if not full: valid=valid[::max(1,len(valid)//128)]
        correct=0; ce=0.
        for start in range(0,len(valid),96):
            batch=store.pack(name,valid[start:start+96],[],device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                h=model.encode(batch['patches'])
                logits=model.parents(h,batch['source'],batch['target'],batch['delta'],batch['mask'])
            correct+=int((logits.argmax(-1)==batch['labels']).sum())
            ce+=float(F.cross_entropy(logits.float(),batch['labels'],reduction='sum'))
        n=len(targets) if full else len(valid)
        nearest=int((g['labels'][valid]==0).sum())
        rows.append(dict(name=name,n=n,covered=len(valid),correct=correct,nearest_correct=nearest,ce_sum=ce))
    return dict(summary=summarize_predictions(rows),videos=rows)


@torch.inference_mode()
def validate_divisions(model,store,names,device,full=True):
    model.eval(); labels=[]; scores=[]
    for name in names:
        g=store.graphs[name]; rows=np.arange(len(g['triples']))
        if not full:
            pos=rows[g['triple_labels']==1]; neg=rows[g['triple_labels']==0]
            rows=np.r_[pos,neg[::max(1,len(neg)//128)]]
        for start in range(0,len(rows),96):
            batch=store.pack(name,[0],[(name,int(i)) for i in rows[start:start+96]],device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                h=model.encode(batch['patches'])
                score=model.divisions(h,batch['triples'],batch['triple_delta'])
            scores.extend(score.float().cpu().tolist()); labels.extend(batch['triple_labels'].cpu().tolist())
    y=np.asarray(labels); s=np.asarray(scores)
    positive=int(y.sum()); negative=len(y)-positive
    if not positive or not negative:
        return dict(positive=positive,negative=negative,balanced_bce=None,average_precision=None)
    loss=np.logaddexp(0,s)-s*y
    order=np.argsort(-s,kind='stable'); sorted_y=y[order]
    precision=np.cumsum(sorted_y)/np.arange(1,len(y)+1)
    return dict(positive=positive,negative=negative,
        balanced_bce=float(.5*(loss[y==1].mean()+loss[y==0].mean())),
        average_precision=float((precision*sorted_y).sum()/positive),
        prevalence=positive/len(y))


def train_fold(root,output,held_group,steps=3000,initial_checkpoint=None):
    config=TemporalConfig(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2 if device.type=='cuda' else 4)
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    store=CacheStore(root); split=make_split(store.names,held_group)
    (output/'split.json').write_text(json.dumps(split,indent=2))
    model=TemporalLinker(config).to(device)
    initialization='random; no public checkpoint';initial_sha=None
    if initial_checkpoint is not None:
        initial=torch.load(initial_checkpoint,map_location=device,weights_only=False)
        if initial['held_group']!=held_group or TemporalConfig(**initial['config'])!=config:
            raise ValueError('Pretraining split/config mismatch')
        if not set(initial['training_videos']).issubset(split['train']):
            raise ValueError('Pretraining used videos outside the fine-tuning training split')
        if any(x['video'] not in split['train'] for x in initial['template_sources']):
            raise ValueError('Pretraining appearance source outside training split')
        model.load_state_dict(initial['state_dict'])
        initialization='Dense procedural lineage pretraining; training-embryo appearance only'
        initial_sha=hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=2e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    rng=np.random.default_rng(config.seed)
    targets={n:np.flatnonzero(store.graphs[n]['labels']>=0) for n in split['train']}
    names=[n for n in split['train'] if len(targets[n])]
    weights=np.array([len(targets[n]) for n in names],float); weights/=weights.sum()
    pos=[]; neg=[]
    for n in names:
        labels=store.graphs[n]['triple_labels']
        pos.extend((n,int(i)) for i in np.flatnonzero(labels==1))
        neg.extend((n,int(i)) for i in np.flatnonzero(labels==0))
    if not pos or not neg: raise ValueError('Need known positive and negative division pairs')
    division_prior=len(pos)/(len(pos)+len(neg))
    started=time.monotonic(); history=[]; best=float('inf')
    for step in range(1,steps+1):
        model.train(); name=str(rng.choice(names,p=weights))
        selected=rng.choice(targets[name],size=min(96,len(targets[name])),replace=False)
        triple_refs=[pos[i] for i in rng.integers(len(pos),size=16)]+[neg[i] for i in rng.integers(len(neg),size=16)]
        batch=store.pack(name,selected,triple_refs,device,drop_parent=True,rng=rng,augment=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
            h=model.encode(batch['patches'])
            logits=model.parents(h,batch['source'],batch['target'],batch['delta'],batch['mask'])
            div=model.divisions(h,batch['triples'],batch['triple_delta'])
            link_loss=F.cross_entropy(logits.float(),batch['labels'])
            division_loss=F.binary_cross_entropy_with_logits(div.float(),batch['triple_labels'])
            loss=link_loss+.25*division_loss
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt); scaler.update(); scheduler.step()
        if step%50==0:
            row=dict(step=step,loss=float(loss),link_loss=float(link_loss),division_loss=float(division_loss),seconds=time.monotonic()-started)
            history.append(row); print('TRAIN',held_group,json.dumps(row),flush=True)
        if step%500==0 or step==steps:
            dev=validate(model,store,split['dev'],device,config,full=False)
            div_dev=validate_divisions(model,store,split['dev'],device,full=False)
            dev['divisions']=div_dev
            value=dev['summary']['ce']+.25*(div_dev['balanced_bce'] or 0.)
            print('DEV',held_group,step,json.dumps(dev['summary']),flush=True)
            if value<best:
                best=value
                torch.save(dict(state_dict=model.state_dict(),config=asdict(config),split=split,step=step,
                    division_prior=division_prior,division_positive_count=len(pos),division_negative_count=len(neg),
                    dev=dev,training_steps_requested=steps,initialization=initialization,
                    initial_checkpoint_sha256=initial_sha),output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2))
    checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    holdout=validate(model,store,split['holdout'],device,config,full=True)
    holdout['divisions']=validate_divisions(model,store,split['holdout'],device,full=True)
    result=dict(held_group=held_group,steps=steps,best_step=checkpoint['step'],seconds=time.monotonic()-started,
        holdout=holdout,scope='Held-out-prefix association test on GT centers plus image-derived distractors; not competition score',
        device=str(device),checkpoint_sha256=hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
        initialization=initialization,initial_checkpoint_sha256=initial_sha)
    (output/'training_receipt.json').write_text(json.dumps(result,indent=2))
    print('FOLD_COMPLETE',held_group,json.dumps(result['holdout']['summary']),flush=True)
