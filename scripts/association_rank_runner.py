"""Train E014 from the completed E013 cache; no image inference or evaluation GT."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from biohub_lab.association_rank import (AssociationRanker, SCALE, node_features,
    competing_groups, sample_rows, ranking_loss, evaluate_ranking)

CONFIG = dict(seed=20260918, steps=6000, learning_rate=3e-4, weight_decay=.01,
              batch=512, ranking_pairs=128, validate_every=500, max_grad_norm=1.)
FROZEN_SHA = '1c8f7d3d1134db7ed3d80355a3a61eac8b99bf2dd9dec68fc50362d16b6fb5c0'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, data):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def load_known(cache, name, record):
    folder = cache/'videos'/name
    for filename, digest in record['files'].items():
        if sha(folder/filename) != digest:
            raise ValueError('Cache checksum mismatch: '+name+'/'+filename)
    graph = arrays(folder/'graph.npz')
    features = arrays(folder/'features.npz')
    nodes = node_features(graph, features['visual'])
    labels = arrays(folder/'labels.npz'); report = read(folder/'labels.json')
    if report['graph_sha256'] != record['files']['graph.npz']:
        raise ValueError('Labels do not belong to this graph')
    result = dict(nodes=nodes, positions=graph['coords'][:, 1:].astype(np.float32)*SCALE/20., heads={})
    for head, key, ykey in [('edge', 'edges', 'edge_y'), ('division', 'triples', 'triple_y')]:
        indices = labels[key]; y = labels[ykey]
        if not np.isin(y, (0,1)).all() or len(indices) != len(y):
            raise ValueError('Unknown or inconsistent label included')
        if len(indices) and (indices.min() < 0 or indices.max() >= len(nodes)):
            raise ValueError('Invalid label node indices')
        groups = competing_groups(indices, y)
        result['heads'][head] = dict(indices=indices, y=y, groups=groups)
    return result


def forward(model, video, indices, device):
    nodes = torch.as_tensor(video['nodes'][indices], dtype=torch.float32, device=device)
    positions = torch.as_tensor(video['positions'][indices], dtype=torch.float32, device=device)
    return model(nodes, positions)


@torch.inference_mode()
def validate(model, videos, head, device):
    model.eval(); reports = {}; saved = {}
    for name, video in videos.items():
        data = video['heads'][head]; indices = data['indices']; parts=[]
        for first in range(0, len(indices), 4096):
            parts.append(forward(model, video, indices[first:first+4096], device).cpu().numpy())
        scores = np.concatenate(parts) if parts else np.empty(0, np.float32)
        if not len(scores):
            continue
        reports[name] = evaluate_ranking(scores, data['y'], data['groups'])
        saved[name] = dict(scores=scores, labels=data['y'])
    valid = [r for r in reports.values() if r['average_precision'] is not None]
    if not valid:
        raise ValueError('No calibration video contains both known classes')
    # Equal weight per video, normalized against the all-tied prevalence AP.
    skill = float(np.mean([(r['average_precision']-r['positive_fraction']) /
                          (1-r['positive_fraction']) for r in valid]))
    return dict(selection_skill=skill, eligible_videos=len(valid), per_video=reports), saved


def train(fit, calibration, root, config=None, device='cuda', data_identity=None):
    config = dict(CONFIG if config is None else config)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    models = {key:AssociationRanker(division=key=='division').to(device) for key in ('edge','division')}
    optimizers = {key:torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay']) for key, model in models.items()}
    schedulers = {key:torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['steps']) for key,opt in optimizers.items()}
    eligible = {key:{
        'positive':[name for name,v in fit.items() if (v['heads'][key]['y']==1).any()],
        'negative':[name for name,v in fit.items() if (v['heads'][key]['y']==0).any()],
        'ranking':[name for name,v in fit.items() if v['heads'][key]['groups']]
        } for key in models}
    if any(not pools['positive'] or not pools['negative'] for pools in eligible.values()):
        raise ValueError('Insufficient known positive and negative training examples')
    best = {key:-float('inf') for key in models}; history=[]; started=time.monotonic()
    save(root/'training_config.json', dict(config=config, eligible_fit_videos=eligible,
         data_identity=data_identity, evaluation_labels_read=False))
    for step in range(1,config['steps']+1):
        losses={}
        for key, model in models.items():
            model.train(); parts=[]; targets=[]
            for label,pool_name in [(1,'positive'),(0,'negative')]:
                pool=eligible[key][pool_name];name=pool[int(rng.integers(len(pool)))];video=fit[name]
                data=video['heads'][key]
                ids=rng.choice(np.flatnonzero(data['y']==label),config['batch']//2)
                parts.append(forward(model,video,data['indices'][ids],device))
                targets.append(torch.full((len(ids),),float(label),device=device))
            pair_logits=torch.empty((0,2),device=device)
            pool=eligible[key]['ranking']
            if pool:
                video=fit[pool[int(rng.integers(len(pool)))]];data=video['heads'][key]
                _,pairs=sample_rows(data['y'],data['groups'],rng,2,config['ranking_pairs'])
                pair_logits=forward(model,video,data['indices'][pairs.ravel()],device).reshape(-1,2)
            loss=ranking_loss(torch.cat(parts),torch.cat(targets),pair_logits)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizers[key].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
            optimizers[key].step();schedulers[key].step();losses[key]=float(loss.detach())
        if step%config['validate_every']==0 or step==config['steps']:
            metrics={}
            for key,model in models.items():
                metrics[key],_=validate(model,calibration,key,device)
                if metrics[key]['selection_skill'] > best[key]:
                    best[key]=metrics[key]['selection_skill']
                    torch.save(dict(state_dict=model.state_dict(),model_config=model.config,step=step,
                        config=config,data_identity=data_identity,validation=metrics[key]),root/f'best_{key}.pt')
            history.append(dict(step=step,losses=losses,
                validation={key:{k:v for k,v in m.items() if k!='per_video'} for key,m in metrics.items()},
                seconds=time.monotonic()-started))
            torch.save(dict(step=step,config=config,data_identity=data_identity,
                models={key:m.state_dict() for key,m in models.items()},
                optimizers={key:o.state_dict() for key,o in optimizers.items()},
                schedulers={key:s.state_dict() for key,s in schedulers.items()},
                torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],best=best),root/'last.pt')
            save(root/'training.json',dict(status='complete' if step==config['steps'] else 'training',
                config=config,history=history,evaluation_labels_read=False))
            print('ASSOCIATION_TRAIN',json.dumps(history[-1]),flush=True)
    selected={}
    for key,model in models.items():
        checkpoint=torch.load(root/f'best_{key}.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        metrics,scores=validate(model,calibration,key,device)
        for name,data in scores.items():
            dest=root/'calibration'/name;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{key}_known_scores.npz',**data)
        selected[key]=dict(step=checkpoint['step'],checkpoint_sha256=sha(root/f'best_{key}.pt'),**metrics)
    save(root/'calibration.json',selected)
    return selected


def main(package, input_root=Path('/kaggle/input'), root=Path('/kaggle/working/association_rank_experiment')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E014',status='verifying_cache',neural_training=True,
        leaderboard_submitted=False,evaluation_labels_read=False,
        scope='Known-candidate calibration only. Complete trajectory evaluation remains pending; conditional development, not independent validation.')
    save(root/'result.json',result)
    try:
        if not torch.cuda.is_available():raise RuntimeError('Authorized E014 training requires a Kaggle GPU')
        matches=[p for p in input_root.rglob('frozen_inputs.json') if sha(p)==FROZEN_SHA]
        if len(matches)!=1:raise ValueError('Expected exactly one pinned completed E013 cache')
        cache=matches[0].parent;frozen=read(matches[0]);split=read(package/'baseline/event_graph_split.json')['split']
        expected=set(split['fit']+split['calibration']+split['evaluation'])
        if set(frozen['records'])!=expected or frozen['annotations_read'] is not False:
            raise ValueError('Frozen input cohort changed')
        original=read(cache/'result.json')
        if original['status']!='complete' or original['split']!=split:
            raise ValueError('Original run did not complete with the required split')
        pins={};loaded={}
        for name in split['fit']+split['calibration']:
            loaded[name]=load_known(cache,name,frozen['records'][name])
            pins[name]=dict(**frozen['records'][name]['files'],labels_sha256=sha(cache/'videos'/name/'labels.npz'))
            print('ASSOCIATION_CACHE_READY',name,flush=True)
        save(root/'input_identity.json',dict(frozen_inputs_sha256=FROZEN_SHA,videos=pins,split=split))
        identity=sha(root/'input_identity.json')
        result.update(status='training',input_identity_sha256=identity,
            gpu=torch.cuda.get_device_name(0),fit_videos=len(split['fit']),calibration_videos=len(split['calibration']))
        save(root/'result.json',result)
        selected=train({n:loaded[n] for n in split['fit']},{n:loaded[n] for n in split['calibration']},root,data_identity=identity)
        result.update(status='complete',steps=CONFIG['steps'],heads={k:{a:b for a,b in v.items() if a!='per_video'} for k,v in selected.items()},
            seconds=time.monotonic()-start,next_action='Inspect calibration ranking, then test structured assignment on complete calibration trajectories before evaluation or submission.')
        save(root/'result.json',result);print('ASSOCIATION_RESULT',json.dumps(result),flush=True)
    except Exception as error:
        result.update(failed_stage=result['status'],status='failed',error=str(error))
        save(root/'result.json',result);(root/'error.txt').write_text(traceback.format_exc());raise
