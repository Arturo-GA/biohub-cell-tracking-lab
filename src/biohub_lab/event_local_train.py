"""E013 training with bounded video loading and restart at optimizer boundaries."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .event_data import save_json
from .event_model import EventGraphNet
from .event_portable import VideoCache, atomic_torch_save, input_identity, restore_rng, rng_state, sha, signature
from .event_train import TRAIN_CONFIG, head_scores, loss_for, positions, threshold, window_batch


EXECUTION = dict(cache_videos=1, attention_chunk=256, checkpoint_activations=True,
                 checkpoint_every=100, checkpoint_seconds=300., cpu_threads=2)


def windows(cache, split):
    regular, dividing, validation = [], [], []
    for role in ('fit', 'calibration'):
        for name in split[role]:
            video = cache[name]; coords = video['graph']['coords']; labels = video['labels']
            frames = np.unique(coords[labels['edges'][:, 0], 0])
            positive = labels['triples'][labels['triple_y'] == 1]
            division_frames = np.unique(coords[positive[:, 0], 0])
            if role == 'fit':
                regular.extend((name, int(t)) for t in frames)
                dividing.extend((name, int(t)) for t in division_frames)
            else:
                if len(frames):
                    validation.append((name, int(frames[len(frames)//2])))
                validation.extend((name, int(t)) for t in division_frames)
            del video, coords, labels, positive
    cache.clear()
    if not regular or not dividing or not validation:
        raise ValueError('Insufficient supervised training/calibration windows')
    return regular, dividing, sorted(set(validation))


def train(root, split, device='cuda', config=None, execution=None, resume=False,
          stop_after=None, time_budget=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    config = dict(TRAIN_CONFIG if config is None else config)
    execution = dict(EXECUTION, **(execution or {})); device = torch.device(device)
    groups = [set(split[k]) for k in ('fit', 'calibration', 'evaluation')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Selector split overlap')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use the dedicated Biohub environment')
    if config['steps'] < 1 or execution['checkpoint_every'] < 1 or execution['attention_chunk'] < 1:
        raise ValueError('Steps and chunk sizes must be positive')
    if (root/'resume.pt').exists() and not resume:
        raise ValueError('Existing training state: explicitly resume or use another output folder')
    if not resume and any((root/name).exists() for name in ('best.pt', 'last.pt')):
        raise ValueError('Existing model weights: preserve them and use a new training folder')
    torch.set_num_threads(execution['cpu_threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    identity = dict(config=config, split=split, inputs_sha256=input_identity(root, split),
                    model=dict(width=64, layers=4), device_type=device.type, torch=str(torch.__version__),
                    attention_chunk=execution['attention_chunk'], checkpoint_activations=execution['checkpoint_activations'])
    identity_hash = signature(identity)
    cache = VideoCache(root, execution['cache_videos']); regular, dividing, validation = windows(cache, split)
    model = EventGraphNet().to(device)
    model.attention_chunk = execution['attention_chunk']; model.checkpoint_activations = execution['checkpoint_activations']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['steps'])
    amp = device.type == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best = float('inf'); history = []; trace = []; first_step = 1
    if resume:
        saved = torch.load(root/'resume.pt', map_location='cpu', weights_only=False)
        if saved.get('format') != 'biohub-resume-v1' or saved['identity_sha256'] != identity_hash:
            raise ValueError('Resume identity mismatch; legacy weights are not an exact restart')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        best = saved['best_loss']; history = saved['history']; trace = saved['trace']; first_step = saved['step']+1
        if saved['best_checkpoint_sha256'] is not None and (
                not (root/'best.pt').is_file() or sha(root/'best.pt') != saved['best_checkpoint_sha256']):
            raise ValueError('Selected checkpoint missing or changed')
        restore_rng(saved['rng'], rng)
    started = last_saved = time.monotonic(); step = first_step-1
    save_json(root/'local_training_identity.json', dict(identity, identity_sha256=identity_hash))

    def persist(status):
        atomic_torch_save(dict(format='biohub-resume-v1', identity_sha256=identity_hash,
            state_dict=model.state_dict(), model_config=model.config, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), step=step,
            rng=rng_state(rng), best_loss=best, history=history, trace=trace,
            best_checkpoint_sha256=sha(root/'best.pt') if (root/'best.pt').is_file() else None),
            root/'resume.pt', backup=True)
        save_json(root/'local_training.json', dict(status=status, step=step, total_steps=config['steps'],
            history=history, identity_sha256=identity_hash, evaluation_labels_read=False,
            checkpoint_sha256=sha(root/'resume.pt'), execution=execution))

    stop_at = min(config['steps'], stop_after if stop_after is not None else config['steps'])
    if stop_at < first_step:
        return dict(status='complete' if step == config['steps'] else 'paused', step=step)
    for step in range(first_step, stop_at+1):
        model.train(); pool = dividing if rng.random() < .5 else regular
        name, frame = pool[int(rng.integers(len(pool)))]; batch = window_batch(cache[name], frame, device, rng, config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = loss_for(model, batch)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss; previous complete checkpoint preserved')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_value = float(loss.detach()); trace.append(dict(step=step, video=name, frame=frame, loss=loss_value))
        del loss, batch
        if step % config['validate_every'] == 0 or step == config['steps']:
            model.eval(); values = []; val_rng = np.random.default_rng(config['seed']+1)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                for val_name, val_frame in validation:
                    values.append(float(loss_for(model, window_batch(cache[val_name], val_frame, device, val_rng, config))))
            value = float(np.mean(values)); improved = value < best
            checkpoint = dict(state_dict=model.state_dict(), model_config=model.config, train_config=config,
                              step=step, validation_loss=value, split=split, identity_sha256=identity_hash)
            atomic_torch_save(checkpoint, root/'last.pt')
            if improved:
                best = value; atomic_torch_save(checkpoint, root/'best.pt')
            history.append(dict(step=step, training_loss=loss_value, calibration_window_loss=value, best=improved))
            print('LOCAL_EVENT_VALIDATION', json.dumps(history[-1]), flush=True)
        now = time.monotonic()
        stop_requested = (root/'STOP').exists() or (time_budget is not None and now-started >= time_budget)
        finished = step == config['steps']; paused = stop_requested or step == stop_at
        if step % execution['checkpoint_every'] == 0 or now-last_saved >= execution['checkpoint_seconds'] or paused:
            persist('complete' if finished else 'paused' if paused else 'training'); last_saved = now
            print('LOCAL_EVENT_CHECKPOINT', step, loss_value, flush=True)
        if stop_requested:
            break
    cache.clear()
    return dict(status='complete' if step == config['steps'] else 'paused', step=step,
                session_seconds=time.monotonic()-started)


def calibrate(root, split, device='cuda', chunk=256):
    from .event_inference import load_model
    root = Path(root)
    progress = json.loads((root/'local_training.json').read_text())
    identity = json.loads((root/'local_training_identity.json').read_text())
    if progress['status'] != 'complete' or identity['split'] != split or identity['inputs_sha256'] != input_identity(root, split):
        raise ValueError('Calibration requires completed training and unchanged split/inputs')
    model = load_model(root/'best.pt', device); cache = VideoCache(root, 1)
    accumulated = {key: [] for key in ('edge_scores', 'edge_y', 'pair_scores', 'pair_y')}
    with torch.inference_mode():
        for name in split['calibration']:
            video = cache[name]; pos = positions(video['graph']['coords'])
            h = model.encode_streamed(video['inputs'], pos, video['neighbors'], device, chunk).to(device)
            p = torch.as_tensor(pos, device=device)
            accumulated['edge_scores'].append(head_scores(model, h, p, video['labels']['edges'], model.links))
            accumulated['edge_y'].append(np.array(video['labels']['edge_y']))
            accumulated['pair_scores'].append(head_scores(model, h, p, video['labels']['triples'], model.divisions))
            accumulated['pair_y'].append(np.array(video['labels']['triple_y']))
            del video, h, p
    joined = {k: np.concatenate(v) for k, v in accumulated.items()}
    np.savez_compressed(root/'calibration_scores.npz', **joined)
    result = dict(edge=threshold(joined['edge_scores'], joined['edge_y'], 1.),
        division=threshold(joined['pair_scores'], joined['pair_y'], .5),
        checkpoint_sha256=sha(root/'best.pt'), scope='Known annotated candidates; calibration videos only')
    save_json(root/'calibration.json', result)
    return result
