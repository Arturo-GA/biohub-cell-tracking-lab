"""Score every E012 eligible pair, then prune by learned score for optimization."""
import csv
from pathlib import Path
import numpy as np
import torch
from .event_data import candidates, save_json
from .event_model import EventGraphNet
from .event_train import load_video, tensor_video, head_scores
from .event_solver import select_graph
from .submission import COLUMNS


def top_per_mother(indices, gains, count):
    if not len(indices):
        return indices, gains
    order = np.lexsort((np.arange(len(gains)), -gains, indices[:, 0]))
    mother = indices[order, 0]
    starts = np.maximum.accumulate(np.where(np.r_[True, mother[1:] != mother[:-1]], np.arange(len(order)), 0))
    keep = order[np.arange(len(order))-starts < count]
    return indices[keep], gains[keep]


def score_graph(model, video, calibration, device='cuda'):
    graph = video['graph']; x, p, neighbors = tensor_video(video, device)
    edges_out, edge_gains, triples_out, triple_gains = [], [], [], []
    scored_edges = scored_pairs = 0
    with torch.inference_mode():
        h = model.encode(x, p, neighbors); quality = model.quality(h).sigmoid().cpu().numpy()
        for first in range(0, len(x), 256):
            mothers = np.arange(first, min(first+256, len(x)))
            edges, triples = candidates(graph, mothers)
            logits = head_scores(model, h, p, edges, model.links)
            pair_logits = head_scores(model, h, p, triples, model.divisions)
            scored_edges += len(edges); scored_pairs += len(triples)
            # Dense mother/slot table lets divisions reuse their two link scores.
            values = np.full((len(mothers), 16), -np.inf, np.float32)
            rr, cc = np.nonzero(graph['daughters'][mothers] >= 0); values[rr, cc] = logits
            if len(triples):
                allowed = np.unpackbits(graph['pair_bits'][mothers], axis=1, bitorder='little')[:, :len(graph['pair_slots'])]
                rows, slots = np.nonzero(allowed); pair = graph['pair_slots'][slots]
                # Mean link gain keeps adding a daughter from creating an automatic reward.
                pair_gain = .5*(values[rows, pair[:, 0]]+values[rows, pair[:, 1]])-calibration['edge']['logit']
                pair_gain += 2*(pair_logits-calibration['division']['logit'])
                valid = pair_logits >= calibration['division']['logit']
                kept, gains = top_per_mother(triples[valid], pair_gain[valid], 2)
                triples_out.append(kept); triple_gains.append(gains)
            if len(edges):
                valid = (graph['continuation'][edges[:, 0]] == edges[:, 1, None]).any(1)
                kept, gains = top_per_mother(edges[valid], logits[valid]-calibration['edge']['logit'], 4)
                edges_out.append(kept); edge_gains.append(gains)
    def cat(parts, width=None):
        return np.concatenate(parts) if parts else np.empty((0, width), np.int64) if width else np.empty(0, np.float32)
    if scored_pairs != int(graph['pair_counts'].sum()):
        raise ValueError('Not all packed division candidates were scored')
    return dict(edges=cat(edges_out, 2), edge_gains=cat(edge_gains), triples=cat(triples_out, 3),
                triple_gains=cat(triple_gains), quality=quality), dict(scored_edges=scored_edges, scored_pairs=scored_pairs,
                optimization_edges=sum(map(len, edges_out)), optimization_pairs=sum(map(len, triples_out)),
                pair_pruning='Calibrated division threshold followed by two highest event gains per mother; no coverage claim after pruning')


def infer_video(root, name, model, calibration, device='cuda'):
    folder = Path(root)/'videos'/name; video = load_video(folder, labels=False)
    scored, counts = score_graph(model, video, calibration, device)
    np.savez_compressed(folder/'event_scores.npz', **scored)
    selected, solver = select_graph(video['graph'], **scored)
    np.savez_compressed(folder/'prediction.npz', coords=video['graph']['coords'], edges=selected,
                        shape=video['graph']['shape'])
    save_json(folder/'prediction.json', dict(counts, solver=solver, evaluation_annotations_read=False))
    return counts


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = EventGraphNet(**saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict']); model.eval()
    return model


def write_csv(root, names, path):
    from .event_train import load_arrays
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(COLUMNS); index = 0
        for name in names:
            pred = load_arrays(Path(root)/'videos'/name/'prediction.npz')
            nodes = np.unique(pred['edges'])
            if not len(nodes):
                raise ValueError('Empty selected graph: '+name)
            for node in nodes:
                writer.writerow([index, name, 'node', int(node), *map(int, pred['coords'][node]), -1, -1]); index += 1
            for a, b in pred['edges']:
                writer.writerow([index, name, 'edge', -1, -1, -1, -1, -1, int(a), int(b)]); index += 1
