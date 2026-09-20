"""Conservative multi-frame bridges between existing track ends and starts."""
import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625])

def propose(ids, coords, edges, dense):
    incoming, outgoing = set(), set()
    for a, b in edges:
        outgoing.add(int(a)); incoming.add(int(b))
    starts = [i for i, n in enumerate(ids) if int(n) not in incoming]
    ends = [i for i, n in enumerate(ids) if int(n) not in outgoing]
    xyz, dxyz = coords[:, 1:] * SCALE, dense[:, 1:] * SCALE
    frames = {}
    for t in np.unique(dense[:, 0]).astype(int):
        di = np.flatnonzero(dense[:, 0] == t)
        hi = np.flatnonzero(coords[:, 0] == t)
        if len(hi):
            di = di[cKDTree(xyz[hi]).query(dxyz[di])[0] > 3.0]
        frames[t] = di
    candidates = []
    start_frames = {}
    for t in np.unique(coords[starts, 0]).astype(int):
        si = np.array([i for i in starts if coords[i, 0] == t], dtype=int)
        start_frames[t] = (si, cKDTree(xyz[si]))
    for a in ends:
        nearby = []
        for dt in range(2, 5):
            entry = start_frames.get(int(coords[a, 0]) + dt)
            if entry is not None:
                si, tree = entry
                nearby.extend(si[tree.query_ball_point(xyz[a], 8 * dt)])
        for b in nearby:
            dt = int(coords[b, 0] - coords[a, 0])
            if not 2 <= dt <= 4 or np.linalg.norm(xyz[b] - xyz[a]) > 8 * dt:
                continue
            donors, residual = [], 0.
            for step in range(1, dt):
                di = frames.get(int(coords[a, 0]) + step, np.array([], dtype=int))
                if not len(di): break
                target = xyz[a] + (xyz[b] - xyz[a]) * step / dt
                distances = np.linalg.norm(dxyz[di] - target, axis=1)
                order = np.argsort(distances)
                if distances[order[0]] > 4: break
                if len(order) > 1 and distances[order[1]] - distances[order[0]] < 1: break
                donors.append(int(di[order[0]])); residual += float(distances[order[0]])
            if len(donors) != dt - 1: continue
            path = np.vstack([xyz[a], dxyz[donors], xyz[b]])
            if np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) > 8: continue
            candidates.append(dict(a=int(a), b=int(b), donors=donors,
                                   cost=residual / (dt - 1) + float(np.linalg.norm(xyz[b]-xyz[a])) / dt))
    # Reject competing endpoint explanations, rather than silently choosing one.
    counts = {}
    for p in candidates:
        for key in [('a', p['a']), ('b', p['b'])]: counts[key] = counts.get(key, 0) + 1
    return [p for p in candidates if counts['a', p['a']] == counts['b', p['b']] == 1]

def augment(ids, coords, edges, dense, proposals, h=None, dh=None):
    new_ids, new_coords, new_edges = list(map(int, ids)), list(coords), list(map(tuple, edges))
    used, accepted, next_id = set(), [], int(max(ids)) + 1
    for p in sorted(proposals, key=lambda p: p['cost']):
        if used.intersection(p['donors']): continue
        if h is not None:
            f = np.vstack([h[p['a']], dh[p['donors']], h[p['b']]])
            if float(np.min(np.sum(f[:-1] * f[1:], axis=1))) < .8: continue
        chain = [int(ids[p['a']])]
        for d in p['donors']:
            new_ids.append(next_id); new_coords.append(dense[d]); chain.append(next_id); next_id += 1
        chain.append(int(ids[p['b']]))
        new_edges.extend(zip(chain[:-1], chain[1:])); used.update(p['donors']); accepted.append(p)
    return np.asarray(new_ids), np.asarray(new_coords), np.asarray(new_edges), dict(bridges=len(accepted), added_nodes=len(new_ids)-len(ids), added_edges=len(new_edges)-len(edges))
