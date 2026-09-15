"""Portable, bounded event data and complete training restart state."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from .event_data import node_inputs, save_json
from .event_train import load_arrays, load_video


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_mmap(folder, labels=True):
    """Unpack one verified video into portable NPY arrays, then publish a receipt."""
    folder = Path(folder); target = folder/'arrays'; target.mkdir(exist_ok=True)
    files = ['graph.npz', 'features.npz'] + (['labels.npz'] if labels else [])
    inputs = {name: sha(folder/name) for name in files}
    receipt_path = target/'manifest.json'
    if receipt_path.is_file():
        old = json.loads(receipt_path.read_text())
        if old['sources'] == inputs and all(sha(target/p) == d for p, d in old['files'].items()):
            return old
        raise ValueError('Existing portable arrays differ from their source/receipt')
    outputs = {}
    for filename in files:
        with np.load(folder/filename, allow_pickle=False) as source:
            for key in source.files:
                relative = filename[:-4]+'_'+key+'.npy'
                temp = target/(relative+'.tmp')
                with temp.open('wb') as handle:
                    np.save(handle, source[key], allow_pickle=False)
                temp.replace(target/relative); outputs[relative] = sha(target/relative)
    receipt = dict(version=1, sources=inputs, files=outputs, labels_included=labels)
    save_json(receipt_path, receipt)
    return receipt


def load_portable(folder, labels=True):
    folder = Path(folder); target = folder/'arrays'
    if not (target/'manifest.json').is_file():
        return load_video(folder, labels=labels)
    receipt = json.loads((target/'manifest.json').read_text())
    if labels and not receipt['labels_included']:
        raise ValueError('Portable video has no supervision')
    def group(prefix):
        return {name[len(prefix)+1:-4]: np.load(target/name, mmap_mode='r', allow_pickle=False)
                for name in receipt['files'] if name.startswith(prefix+'_')}
    graph = group('graph'); features = group('features')
    result = dict(graph=graph, inputs=node_inputs(graph, features['visual']), neighbors=features['neighbors'])
    if labels:
        result['labels'] = group('labels')
    return result


class VideoCache:
    """One shared LRU for fit and calibration; sampling never depends on cache."""
    def __init__(self, root, capacity=1):
        if capacity < 1:
            raise ValueError('Cache capacity must be positive')
        self.root = Path(root); self.capacity = capacity; self.items = OrderedDict()

    def __getitem__(self, name):
        if name not in self.items:
            # Evict before loading to avoid holding capacity+1 whole videos.
            if len(self.items) >= self.capacity:
                self.items.popitem(last=False)
            self.items[name] = load_portable(self.root/'videos'/name)
        self.items.move_to_end(name)
        return self.items[name]

    def clear(self):
        self.items.clear()


def input_identity(root, split):
    """Bind training restart to actual arrays, without reading evaluation labels."""
    files = {}
    for name in split['fit'] + split['calibration']:
        folder = Path(root)/'videos'/name
        manifest = folder/'arrays/manifest.json'
        if manifest.is_file():
            receipt = json.loads(manifest.read_text())
            if not receipt['labels_included']:
                raise ValueError('Missing labels: '+name)
            for relative, expected in receipt['files'].items():
                actual = sha(folder/'arrays'/relative)
                if actual != expected:
                    raise ValueError('Portable array checksum changed: '+name+'/'+relative)
            files[name] = receipt
        else:
            files[name] = {f: sha(folder/f) for f in ('graph.npz', 'features.npz', 'labels.npz')}
    return signature(files)


def atomic_torch_save(value, path, backup=False):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle:
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    if backup and path.is_file():
        shutil.copyfile(path, path.with_suffix(path.suffix+'.previous'))
    temp.replace(path)


def rng_state(rng):
    return dict(sampler=rng.bit_generator.state, python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, rng):
    rng.bit_generator.state = state['sampler']; random.setstate(state['python'])
    np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA resume requires the original device type')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
