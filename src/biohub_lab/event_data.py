"""E013 detector-space supervision and frozen dual-UNet visual features."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from .detector_proposals import SCALE


def select_split(names, development, excluded, seed='biohub-events-v1'):
    if len(names) != len(set(names)):
        raise ValueError('Duplicate video names')
    groups = {}
    for name in sorted(set(names) - set(development) - set(excluded)):
        groups.setdefault(name.split('_')[0], []).append(name)
    fit, calibration = [], []
    if len(groups) != 2:
        raise ValueError('Expected two acquisition groups')
    for values in groups.values():
        ordered = sorted(values, key=lambda n: (hashlib.sha256((seed + ':' + n).encode()).hexdigest(), n))
        if len(ordered) < 32:
            raise ValueError('Insufficient videos')
        fit.extend(ordered[:24]); calibration.extend(ordered[24:32])
    return dict(seed=seed, fit=sorted(fit), calibration=sorted(calibration), evaluation=sorted(development),
                excluded=sorted(excluded), reserved=sorted(set(names)-set(fit)-set(calibration)-set(development)-set(excluded)))


def neighborhoods(coords, k=8):
    """Self plus k spatial neighbors in each of the previous/current/next frames."""
    result = np.full((len(coords), 1 + 3*k), -1, np.int32)
    result[:, 0] = np.arange(len(coords))
    for t in np.unique(coords[:, 0]):
        ids = np.flatnonzero(coords[:, 0] == t)
        for j, dt in enumerate((-1, 0, 1)):
            other = np.flatnonzero(coords[:, 0] == t+dt)
            if not len(other):
                continue
            d, ix = cKDTree(coords[other, 1:]*SCALE).query(coords[ids, 1:]*SCALE, k=k,
                                                         distance_upper_bound=20.)
            valid = np.isfinite(d); rows, cols = np.nonzero(valid)
            result[ids[rows], 1+j*k+cols] = other[ix[rows, cols]]
    return result


def node_inputs(graph, visual):
    if visual.shape != (len(graph['coords']), 64) or not np.isfinite(visual).all():
        raise ValueError('Invalid dual-UNet features')
    source = np.eye(3, dtype=np.float32)[graph['origin']]
    position = graph['coords'] / np.maximum(graph['shape']-1, 1)
    return np.concatenate((visual.astype(np.float32), source,
                           np.tanh(graph['scores'][:, None]), position), axis=1).astype(np.float32)


def extract_visual(module, models, image_path, coords, device):
    """First-seen two-frame windows, single view, subvoxel feature sampling."""
    import torch
    import torch.nn.functional as F
    import zarr
    from .harmonic_centers import image_metadata
    meta = image_metadata(image_path); group = zarr.open_group(str(image_path), mode='r')
    arr = group['0']; shape = list(meta.image_shape[1:]); ds = (1, 4, 4)
    q = meta.quantiles
    low, high = float(q.get('0.001', 0.)), float(q.get('0.999', 1.))
    if not q or high <= low:
        raise ValueError('Missing or invalid image normalization quantiles')
    if not models:
        raise ValueError('At least one frozen encoder is required')
    result = np.zeros((len(coords), 32*len(models)), np.float16); seen = np.zeros(len(coords), bool)
    if arr.shape[0] < 2:
        raise ValueError('Two-frame encoders require at least two frames')
    with torch.inference_mode():
        for start in range(arr.shape[0]-1):
            frames = [module._load_frame(arr, t, shape, ds) for t in (start, start+1)]
            imgs = torch.stack(frames).to(device=device, dtype=torch.float32)
            imgs = ((imgs-low)/(high-low+1e-6)).clamp_min(0).unsqueeze(0)
            wanted = (0, 1) if start == 0 else (1,)
            for m, model in enumerate(models):
                features, _ = model.encode(imgs)
                if features.shape[2] != 32:
                    raise ValueError('Pretrained feature channel count changed')
                for f in wanted:
                    ids = np.flatnonzero(coords[:, 0] == start+f)
                    if not len(ids):
                        continue
                    loc = torch.as_tensor(coords[ids, 1:]/np.array(ds), dtype=torch.float32, device=device)
                    extent = torch.tensor(features.shape[-3:], device=device)-1
                    loc = 2*loc/extent.clamp_min(1)-1
                    grid = loc[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
                    values = F.grid_sample(features[:, f].float(), grid, align_corners=True,
                                           padding_mode='border').reshape(32, -1).T
                    result[ids, 32*m:32*(m+1)] = values.cpu().numpy().astype(np.float16)
                    seen[ids] = True
                del features
    if not seen.all() or not np.isfinite(result).all():
        raise ValueError('Incomplete visual feature extraction')
    return result


def candidates(graph, mothers):
    """Materialize packed pairs only for the requested source nodes."""
    mothers = np.asarray(mothers, np.int64)
    table = graph['daughters'][mothers]
    a, b = np.nonzero(table >= 0)
    edges = np.column_stack((mothers[a], table[a, b])).astype(np.int64)
    allowed = np.unpackbits(graph['pair_bits'][mothers], axis=1, bitorder='little')[:, :len(graph['pair_slots'])]
    rows, slots = np.nonzero(allowed)
    pair = graph['pair_slots'][slots]
    triples = np.column_stack((mothers[rows], table[rows, pair[:, 0]], table[rows, pair[:, 1]])).astype(np.int64)
    return edges, triples


def sparse_labels(graph, truth, gt_edges, match_um=7., ambiguity_um=1.):
    """Supervise actual detections. An unknown label is never a negative label."""
    coords = graph['coords']; mapping = np.full(len(coords), -1, np.int64)
    distance = np.full(len(coords), np.inf, np.float32)
    for t in np.unique(coords[:, 0]):
        ids = np.flatnonzero(coords[:, 0] == t); ref = np.flatnonzero(truth[:, 0] == t)
        if not len(ref):
            continue
        dist, local = cKDTree(truth[ref, 1:]*SCALE).query(coords[ids, 1:]*SCALE, k=2)
        good = (dist[:, 0] <= match_um) & ((dist[:, 1]-dist[:, 0]) >= ambiguity_um)
        mapping[ids[good]] = ref[local[good, 0]]; distance[ids[good]] = dist[good, 0]
    parent = np.full(len(truth), -1, np.int64)
    for s, t in gt_edges:
        if parent[t] >= 0:
            raise ValueError('Annotated target has multiple parents')
        parent[t] = s
    edges_out, edge_y, triples_out, triple_y = [], [], [], []
    unique_events = set()
    for first in range(0, len(coords), 1024):
        mothers = np.arange(first, min(first+1024, len(coords)))
        mothers = mothers[mapping[mothers] >= 0]
        edges, triples = candidates(graph, mothers)
        if len(edges):
            s, t = mapping[edges].T
            known = (t >= 0) & (parent[t.clip(0)] >= 0)
            edges_out.append(edges[known]); edge_y.append((parent[t[known]] == s[known]).astype(np.float32))
        if len(triples):
            m, a, b = mapping[triples].T
            pa, pb = parent[a.clip(0)], parent[b.clip(0)]
            positive = (a >= 0) & (b >= 0) & (a != b) & (pa == m) & (pb == m)
            # A single known incompatible parent disproves the division. Unmapped
            # daughters, or two detections of one child, alone prove nothing.
            negative = ((a >= 0) & (pa >= 0) & (pa != m)) | ((b >= 0) & (pb >= 0) & (pb != m))
            known = positive | negative
            triples_out.append(triples[known]); triple_y.append(positive[known].astype(np.float32))
            unique_events.update((int(x), *sorted((int(y), int(z)))) for x, y, z in zip(m[positive], a[positive], b[positive]))
    quality = np.exp(-.5*(distance/3.5)**2).astype(np.float32)
    def cat(values, width=None):
        return np.concatenate(values) if values else np.empty((0, width), np.int64) if width else np.empty(0, np.float32)
    result = dict(edges=cat(edges_out, 2), edge_y=cat(edge_y), triples=cat(triples_out, 3),
                  triple_y=cat(triple_y), quality=quality, quality_mask=mapping >= 0)
    counts = dict(matched_candidates=int((mapping >= 0).sum()), known_edges=len(result['edges']),
                  positive_edges=int(result['edge_y'].sum()), known_pairs=len(result['triples']),
                  positive_pairs=int(result['triple_y'].sum()), represented_unique_divisions=len(unique_events),
                  annotated_divisions=int((np.bincount(gt_edges[:, 0], minlength=len(truth)) == 2).sum()))
    return result, counts


def save_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)
