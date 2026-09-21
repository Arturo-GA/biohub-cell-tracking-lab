"""Detection agreement and robust local motion for complementary gap proposals.

Local median displacement follows the public motion-flow idea; weights use
observable predictions, never embryo names or annotation membership.
"""
import numpy as np
from scipy.spatial import cKDTree
from .temporal_bridge import SCALE, propose, augment

def combine(ids, coords, edges, dense, consensus, mode):
    if mode == 'dense':
        return augment(ids, coords, edges, dense, propose(ids, coords, edges, dense))
    points = dense.astype(float).copy()
    support = np.zeros(len(dense), dtype=float)
    weights = np.zeros(len(dense), dtype=float)
    for t in np.unique(dense[:, 0]):
        ix = np.flatnonzero(dense[:, 0] == t)
        jx = np.flatnonzero(consensus[:, 0] == t)
        if not len(jx):
            continue
        distance, near = cKDTree(consensus[jx, 1:] * SCALE).query(dense[ix, 1:] * SCALE)
        # Higher peak rank and agreement receive greater weight; dense is the
        # more precise specialist and retains at least 75% of the location.
        rd = 1 / (1 + np.arange(len(ix)) / 128)
        rc = 1 / (1 + near / 128)
        agreement = np.exp(-0.5 * (distance / 1.625) ** 2)
        weight = 0.5 * rc / (rd + rc) * agreement
        weight = np.minimum(weight, 0.25)
        weight[distance > 3] = 0
        support[ix] = agreement
        weights[ix] = weight
        if mode in ('weighted', 'weighted_motion'):
            points[ix, 1:] = (1-weight[:, None])*dense[ix, 1:] + weight[:, None]*consensus[jx[near], 1:]
    points = np.rint(points).astype(np.int64)
    proposals = propose(ids, coords, edges, points)
    lookup = {int(n): i for i, n in enumerate(ids)}
    incoming = {}
    velocities = {}
    for a, b in edges:
        ia, ib = lookup[int(a)], lookup[int(b)]
        dt = coords[ib, 0] - coords[ia, 0]
        if dt == 1:
            incoming[ib] = (coords[ib, 1:] - coords[ia, 1:]) * SCALE
            velocities.setdefault(int(coords[ib, 0]), []).append((ib, incoming[ib]))
    accepted = []
    for p in proposals:
        a, b = p['a'], p['b']
        if mode == 'weighted':
            accepted.append(p)
            continue
        neighbors = velocities.get(int(coords[a, 0]), [])
        if not neighbors:
            continue
        vi = np.array([q[0] for q in neighbors])
        vel = np.array([q[1] for q in neighbors])
        dist = np.linalg.norm((coords[vi, 1:] - coords[a, 1:]) * SCALE, axis=1)
        chosen = np.argsort(dist)[:8]
        chosen = chosen[dist[chosen] <= 20]
        if len(chosen) < 3:
            continue
        flow = np.median(vel[chosen], axis=0)
        spread = max(1.625, float(np.median(np.linalg.norm(vel[chosen]-flow, axis=1))))
        own = incoming.get(a, flow)
        # Trust a coherent track history more than the neighborhood; otherwise
        # let the neighborhood dominate. This is a per-track soft weight.
        own_weight = 0.75 * np.exp(-np.linalg.norm(own-flow)/(2*spread))
        prediction = own_weight*own + (1-own_weight)*flow
        dt = float(coords[b, 0]-coords[a, 0])
        observed = (coords[b, 1:]-coords[a, 1:])*SCALE/dt
        motion_quality = np.exp(-0.5*(np.linalg.norm(observed-prediction)/(2*spread))**2)
        agreement = float(np.mean(support[p['donors']]))
        reliability = 0.75*motion_quality + 0.25*agreement
        if reliability >= 0.65:
            accepted.append(p)
    ns, cs, es, report = augment(ids, coords, edges, points, accepted)
    report.update(proposals=len(proposals), mean_secondary_weight=float(weights.mean()) if len(weights) else 0)
    return ns, cs, es, report
