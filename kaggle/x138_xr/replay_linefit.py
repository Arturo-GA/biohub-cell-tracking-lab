"""Batched upstream linefit and writer rounding, recomputed on each edited graph."""
import collections
import numpy as np


def rounded_node(node):
    return (int(node['t']), *(max(0, int(round(float(node[k])))) for k in ('z', 'y', 'x')))


def full_linefit_round(nodes, edges, window=2, weight=0.8, raw_output=False, tie_weights=()):
    """Batch the reference polyfit by window shape, refitting ties individually.

Recompute every node: capture raw positions are float32 whereas the saved final
positions were computed before that quantization. Never mix these two baselines.
"""
    def encode(v):
        return (int(v['t']), *(float(v[c]) for c in ('z','y','x'))) if raw_output else rounded_node(v)
    if not nodes or not edges or window <= 0 or weight <= 0:
        return {k: encode(v) for k, v in nodes.items()}
    pred, succ = collections.defaultdict(list), collections.defaultdict(list)
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id'])
        if s in nodes and d in nodes and int(nodes[d]['t']) == int(nodes[s]['t']) + 1:
            succ[s].append(d)
            pred[d].append(s)
    groups = collections.defaultdict(list)
    positions = {k: np.array([v[c] for c in ('z', 'y', 'x')], dtype=np.float64) for k, v in nodes.items()}
    out = {k: encode(v) for k, v in nodes.items()}
    for k in sorted(nodes):
        hood = [(0, k)]
        for sign, adj in ((-1, pred), (1, succ)):
            current = k
            for step in range(1, window + 1):
                ids = adj.get(current, [])
                if len(ids) != 1:
                    break
                current = ids[0]
                hood.append((sign * step, current))
        if len(hood) >= 3:
            groups[tuple(dt for dt, _ in hood)].append((k, [i for _, i in hood]))
    weight = float(np.clip(weight, 0.0, 1.0))
    for offsets, members in groups.items():
        dts = np.array(offsets, dtype=np.float64)
        # Bounded RHS size keeps memory predictable on CPU notebooks.
        for start in range(0, len(members), 8192):
            batch = members[start:start + 8192]
            coords = np.array([[positions[n] for n in hood] for _, hood in batch])
            rhs = coords.transpose(1, 0, 2).reshape(len(offsets), -1)
            fits = np.polyfit(dts, rhs, 1)[1].reshape(-1, 3)
            for idx, (k, _) in enumerate(batch):
                fit = fits[idx]
                if not np.isfinite(fit).all():
                    continue
                pos = (1.0 - weight) * positions[k] + weight * fit
                tie_positions = [pos] + [(1-w)*positions[k]+w*fit for w in tie_weights]
                if any(np.any(np.abs(p - np.floor(p) - 0.5) < 1e-7) for p in tie_positions):
                    fit = np.array([np.polyval(np.polyfit(dts, coords[idx, :, axis], 1), 0.0) for axis in range(3)])
                    pos = (1.0 - weight) * positions[k] + weight * fit
                out[k] = (int(nodes[k]['t']), *(float(x) if raw_output else max(0, int(round(float(x)))) for x in pos))
    return out
