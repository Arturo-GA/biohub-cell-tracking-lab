"""E030 CPU: fit small lineage heads on frozen DINO features, then diagnose."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.image_identity import ImageIdentityModel, SCALE


def locate(suffix):
    values = list(Path('/kaggle/input').rglob(suffix)); assert len(values)==1,(suffix,len(values)); return values[0]


def balanced_bce(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    groups = [losses[labels==y].mean() for y in (0,1) if (labels==y).any()]
    return torch.stack(groups).mean() if groups else logits.sum()*0


def rank_loss(model,h,d,rng):
    targets = np.unique(d['edges'][:,1].numpy())
    if len(targets)>32: targets = rng.choice(targets,32,replace=False)
    losses = []
    for target in targets:
        ids = torch.where(d['edges'][:,1]==int(target))[0]
        correct = d['edge_y'][ids]>0
        # Fit-only parent deletion creates supervised absent-parent/null examples.
        if correct.any() and rng.random()<.15: ids = ids[~correct]
        if not len(ids): continue
        scores = model.edge_logits(h,d['coords'],d['edges'][ids]); scores = torch.cat((scores,model.null[None]))
        good = torch.where(d['edge_y'][ids]>0)[0]
        label = int(good[0]) if len(good) else len(ids)
        assert len(good)<=1
        losses.append(F.cross_entropy(scores[None],torch.tensor([label])))
    return torch.stack(losses).mean() if losses else h.sum()*0


def sample_binary(d,name,rng,limit=256):
    y = d[name+'_y']; positive = np.flatnonzero(y.numpy()>0); negative = np.flatnonzero(y.numpy()==0)
    selected = []
    for pool in (positive,negative):
        if len(pool): selected.extend(rng.choice(pool,min(len(pool),limit//2),replace=False).tolist())
    return torch.as_tensor(selected,dtype=torch.long)


def train(rows,config,use_image):
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    model = ImageIdentityModel(config['features'],config['hidden'],use_image)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01)
    losses = []
    for step in range(config['steps']):
        d = rows[int(rng.integers(len(rows)))]; h = model.encode(d['features'])
        loss = rank_loss(model,h,d,rng)
        for name,key,method,weight in [('pair','pairs',model.identity_logits,.5),('triple','triples',model.division_logits,1.)]:
            ids = sample_binary(d,name,rng)
            if len(ids): loss = loss+weight*balanced_bce(method(h,d['coords'],d[key][ids]),d[name+'_y'][ids])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1)%100==0: print('IDENTITY_TRAIN',use_image,step+1,float(np.mean(losses[-100:])),flush=True)
    return model.eval(),dict(first100=float(np.mean(losses[:100])),last100=float(np.mean(losses[-100:])),steps=config['steps'])


def average_precision(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float)
    if not y.any():return None
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    # Aggregate tied scores together, avoiding optimistic ordering of positives.
    end=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[end];precision=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())


def evaluate(model,rows,shuffle=False):
    report=[]; division_y=[];division_s=[];identity_y=[];identity_s=[]
    rng=np.random.default_rng(30030)
    with torch.inference_mode():
        for d in rows:
            features=d['features']
            if shuffle: features=features[torch.as_tensor(rng.permutation(len(features)))]
            h=model.encode(features);scores=model.edge_logits(h,d['coords'],d['edges']).numpy()
            nearest=correct=total=represented=represented_correct=null_correct=0
            for target in np.unique(d['edges'][:,1].numpy()):
                ids=np.flatnonzero(d['edges'][:,1].numpy()==target);y=d['edge_y'][ids].numpy();pos=np.flatnonzero(y>0)
                label=int(pos[0]) if len(pos) else len(ids)
                predicted=int(np.argmax(np.r_[scores[ids],float(model.null)]))
                a,b=d['edges'][ids].T
                distance=np.linalg.norm((d['coords'][a,1:]-d['coords'][b,1:]).numpy()*SCALE,axis=1)
                total+=1;correct+=predicted==label;represented+=bool(len(pos));nearest+=bool(len(pos)) and int(np.argmin(distance))==label
                represented_correct+=bool(len(pos)) and predicted==label
                null_correct+=not len(pos) and predicted==label
            ds=model.division_logits(h,d['coords'],d['triples']).numpy();iscore=model.identity_logits(h,d['coords'],d['pairs']).numpy()
            division_s.extend(ds.tolist());division_y.extend(d['triple_y'].numpy().tolist());identity_s.extend(iscore.tolist());identity_y.extend(d['pair_y'].numpy().tolist())
            report.append(dict(video=d['video'],targets=total,represented=represented,correct=correct,represented_correct=represented_correct,null_correct=null_correct,nearest_correct=nearest,division_positive=int(d['triple_y'].sum())))
    total=sum(r['targets'] for r in report);correct=sum(r['correct'] for r in report);nearest=sum(r['nearest_correct'] for r in report)
    represented=sum(r['represented'] for r in report)
    return dict(videos=report,targets=total,parent_accuracy=correct/max(total,1),nearest_accuracy=nearest/max(total,1),represented_targets=represented,represented_parent_accuracy=sum(r['represented_correct'] for r in report)/max(represented,1),represented_nearest_accuracy=nearest/max(represented,1),null_targets=total-represented,null_correct=sum(r['null_correct'] for r in report),division_ap=average_precision(division_y,division_s),division_positives=int(sum(division_y)),identity_ap=average_precision(identity_y,identity_s),identity_pairs=len(identity_y))


def main(package):
    torch.set_num_threads(2);start=time.monotonic();root=Path('/kaggle/working/image_identity_train');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e030_image_identity.json').read_text())
    data_path=locate('image_identity_data/result.json');feature_path=locate('image_identity_features/result.json')
    data=json.loads(data_path.read_text());features=json.loads(feature_path.read_text())
    assert data['status']==features['status']=='complete' and data['config']==features['config']==config
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==features['input_manifest_sha256']
    lookup={r['video']:r for r in features['videos']};rows={'fit':[],'validation':[]}
    for item in data['videos']:
        p=data_path.parent/item['file'];f=lookup[item['video']];q=feature_path.parent/f['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']==f['source_sha256']
        assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256'] and f['split']==item['split']
        with np.load(p,allow_pickle=False) as a:
            d={key:torch.as_tensor(a[key]) for key in ('coords','edges','edge_y','triples','triple_y','pairs','pair_y')}
        d.update(features=torch.as_tensor(np.load(q,allow_pickle=False).astype(np.float32)),video=item['video'])
        rows[item['split']].append(d)
    assert {r['video'] for r in rows['fit']}==set(config['fit']) and {r['video'] for r in rows['validation']}==set(config['validation'])
    results={};models={}
    for label,use_image in [('image',True),('geometry_ablation',False)]:
        model,history=train(rows['fit'],config,use_image);models[label]=model
        checkpoint=root/(label+'.pt');torch.save(dict(state_dict=model.state_dict(),config=config,use_image=use_image),checkpoint)
        results[label]=dict(training=history,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    # Freeze both checkpoints before evaluating ANY calibration labels.
    for label,model in models.items(): results[label]['validation']=evaluate(model,rows['validation'])
    results['shuffled_image_control']=evaluate(models['image'],rows['validation'],shuffle=True)
    image=results['image']['validation'];geo=results['geometry_ablation']['validation']
    result=dict(status='complete',config=config,results=results,seconds=time.monotonic()-start,
        image_parent_delta_vs_geometry=image['parent_accuracy']-geo['parent_accuracy'],
        scope='Jittered annotation-space diagnostic, NOT detector-space or official graph score',
        leaderboard_submitted=False,automatic_promotion=False,
        input_manifest_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),feature_manifest_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('IDENTITY_TRAIN_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
