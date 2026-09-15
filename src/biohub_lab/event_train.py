"""Fit only the new graph selector; calibrate on disjoint video names."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from .event_data import node_inputs, save_json
from .event_model import EventGraphNet
from .detector_proposals import SCALE

TRAIN_CONFIG = dict(seed=20260915, steps=6000, learning_rate=3e-4, weight_decay=.01,
                    validate_every=500, context_before=4, context_after=5,
                    edge_batch=1024, pair_batch=512, quality_batch=512)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_video(folder, labels=True):
    folder = Path(folder); graph = load_arrays(folder/'graph.npz')
    features = load_arrays(folder/'features.npz')
    result = dict(graph=graph, inputs=node_inputs(graph, features['visual']), neighbors=features['neighbors'])
    if labels:
        result['labels'] = load_arrays(folder/'labels.npz')
    return result


def positions(coords):
    return (coords*np.array([1., *list(SCALE/20.)])).astype(np.float32)


def balanced_indices(y, maximum, rng):
    pos = np.flatnonzero(y > .5); neg = np.flatnonzero(y <= .5)
    take_p = min(len(pos), maximum//2); take_n = min(len(neg), maximum-take_p)
    return np.concatenate((rng.choice(pos, take_p, replace=False), rng.choice(neg, take_n, replace=False)))


def window_batch(video, frame, device, rng, config=TRAIN_CONFIG):
    coords = video['graph']['coords']; labels = video['labels']
    ids = np.flatnonzero((coords[:, 0] >= frame-config['context_before']) &
                         (coords[:, 0] <= frame+config['context_after']))
    remap = np.full(len(coords), -1, np.int64); remap[ids] = np.arange(len(ids))
    neighbors = video['neighbors'][ids]; local = remap[neighbors.clip(0)]; local[neighbors < 0] = -1
    result = dict(inputs=torch.tensor(video['inputs'][ids], device=device),
                  positions=torch.tensor(positions(coords[ids]), device=device),
                  neighbors=torch.tensor(local, device=device), targets={})
    for key, ykey, batch in [('edges', 'edge_y', 'edge_batch'), ('triples', 'triple_y', 'pair_batch')]:
        all_ids = np.flatnonzero(coords[labels[key][:, 0], 0] == frame)
        selected = all_ids[balanced_indices(labels[ykey][all_ids], config[batch], rng)]
        result['targets'][key] = torch.tensor(remap[labels[key][selected]], device=device)
        result['targets'][ykey] = torch.tensor(labels[ykey][selected], device=device)
    qids = ids[(coords[ids, 0] == frame) & labels['quality_mask'][ids]]
    if len(qids) > config['quality_batch']:
        qids = rng.choice(qids, config['quality_batch'], replace=False)
    result['targets']['quality_ids'] = torch.tensor(remap[qids], device=device)
    result['targets']['quality'] = torch.tensor(labels['quality'][qids], device=device)
    return result


def loss_for(model, batch):
    h = model.encode(batch['inputs'], batch['positions'], batch['neighbors']); target = batch['targets']
    loss = h.sum()*0
    for key, ykey, head in [('edges', 'edge_y', model.links), ('triples', 'triple_y', model.divisions)]:
        if len(target[key]):
            logits = head(h, batch['positions'], target[key])
            loss = loss + F.binary_cross_entropy_with_logits(logits, target[ykey])
    if len(target['quality_ids']):
        loss = loss + .25*F.binary_cross_entropy_with_logits(model.quality(h[target['quality_ids']]), target['quality'])
    return loss


def tensor_video(video, device):
    return (torch.tensor(video['inputs'], device=device),
            torch.tensor(positions(video['graph']['coords']), device=device),
            torch.tensor(video['neighbors'].astype(np.int64), device=device))


def head_scores(model, h, p, indices, head, chunk=8192):
    scores = []
    for first in range(0, len(indices), chunk):
        ix = torch.as_tensor(indices[first:first+chunk], dtype=torch.long, device=h.device)
        scores.append(head(h, p, ix).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, np.float32)


def threshold(scores, labels, beta):
    """Known-case F-beta only: unknown detections contribute no pseudo-negatives."""
    scores = np.asarray(scores); labels = np.asarray(labels)
    if not len(scores) or not labels.sum() or labels.sum() == len(labels):
        raise ValueError('Calibration requires positive and negative examples')
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    best = None
    for value in thresholds:
        pred = scores >= value; tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum()); fn = int((~pred & (labels == 1)).sum())
        metric = (1+beta**2)*tp/max((1+beta**2)*tp+beta**2*fn+fp, 1e-9)
        # Conservative tie break. Thresholds are head logits, not probabilities.
        row = (metric, float(value), tp, fp, fn)
        if best is None or row[:2] > best[:2]:
            best = row
    return dict(logit=best[1], f_beta=best[0], beta=beta, tp=best[2], fp=best[3], fn=best[4],
                positives=int(labels.sum()), negatives=int((labels == 0).sum()), candidates=len(scores))


def train_and_calibrate(root, split, device='cuda', config=None):
    root = Path(root); config = dict(TRAIN_CONFIG if config is None else config)
    if set(split['fit']) & set(split['calibration']) or (set(split['fit']) | set(split['calibration'])) & set(split['evaluation']):
        raise ValueError('Selector split overlap')
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); torch.set_num_threads(2)
    rng = np.random.default_rng(config['seed']); device = torch.device(device)
    fit = {n: load_video(root/'videos'/n) for n in split['fit']}
    cal = {n: load_video(root/'videos'/n) for n in split['calibration']}
    regular, dividing, validation = [], [], []
    for name, v in fit.items():
        regular.extend((name, int(t)) for t in np.unique(v['graph']['coords'][v['labels']['edges'][:, 0], 0]))
        positive = v['labels']['triples'][v['labels']['triple_y'] == 1]
        dividing.extend((name, int(t)) for t in np.unique(v['graph']['coords'][positive[:, 0], 0]))
    for name, v in cal.items():
        frames = np.unique(v['graph']['coords'][v['labels']['edges'][:, 0], 0])
        if len(frames):
            validation.append((name, int(frames[len(frames)//2])))
        positive = v['labels']['triples'][v['labels']['triple_y'] == 1]
        validation.extend((name, int(t)) for t in np.unique(v['graph']['coords'][positive[:, 0], 0]))
    if not regular or not dividing or not validation:
        raise ValueError('Insufficient supervised training/calibration windows')
    model = EventGraphNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['steps'])
    amp = device.type == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best = float('inf'); history = []; started = time.monotonic()
    receipt = dict(config=config, model=model.config, fit=split['fit'], calibration=split['calibration'],
                   evaluation_labels_read=False, regular_windows=len(regular), division_windows=len(dividing))
    for step in range(1, config['steps']+1):
        model.train(); pool = dividing if rng.random() < .5 else regular
        name, frame = pool[int(rng.integers(len(pool)))]; batch = window_batch(fit[name], frame, device, rng, config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = loss_for(model, batch)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); scaler.step(optimizer); scaler.update(); scheduler.step()
        if step % config['validate_every'] == 0 or step == config['steps']:
            model.eval(); values = []; val_rng = np.random.default_rng(config['seed']+1)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                for name, frame in sorted(set(validation)):
                    values.append(float(loss_for(model, window_batch(cal[name], frame, device, val_rng, config))))
            value = float(np.mean(values)); improved = value < best
            state = dict(state_dict=model.state_dict(), model_config=model.config, train_config=config,
                         step=step, validation_loss=value, split=split)
            torch.save(state, root/'last.pt')
            if improved:
                best = value; torch.save(state, root/'best.pt')
            history.append(dict(step=step, training_loss=float(loss.detach()), calibration_window_loss=value,
                                best=improved, seconds=time.monotonic()-started))
            save_json(root/'training.json', dict(receipt, status='training', history=history))
            print('EVENT_TRAIN', json.dumps(history[-1]), flush=True)
    del fit
    checkpoint = torch.load(root/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    accumulated = {key: [] for key in ('edge_scores', 'edge_y', 'pair_scores', 'pair_y')}
    with torch.inference_mode():
        for name, v in cal.items():
            x, p, neighbors = tensor_video(v, device); h = model.encode(x, p, neighbors)
            accumulated['edge_scores'].append(head_scores(model, h, p, v['labels']['edges'], model.links))
            accumulated['edge_y'].append(v['labels']['edge_y'])
            accumulated['pair_scores'].append(head_scores(model, h, p, v['labels']['triples'], model.divisions))
            accumulated['pair_y'].append(v['labels']['triple_y'])
            del x, p, neighbors, h
    joined = {k: np.concatenate(v) for k, v in accumulated.items()}
    np.savez_compressed(root/'calibration_scores.npz', **joined)
    calibration = dict(edge=threshold(joined['edge_scores'], joined['edge_y'], 1.),
                       division=threshold(joined['pair_scores'], joined['pair_y'], .5),
                       scope='Known annotated candidates only; sampled detector correspondences are not independent biological events.',
                       checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest())
    save_json(root/'calibration.json', calibration)
    save_json(root/'training.json', dict(receipt, status='complete', history=history,
                                        selected_step=checkpoint['step'], calibration=calibration))
    return calibration
