"""Overlapping temporal MILPs with activation, births, deaths and binary divisions."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from .detector_proposals import SCALE

SOLVER_CONFIG = dict(window=8, stride=4, time_limit=5., gap=.02,
                     birth_cost=.7, death_cost=.7, duplicate_radius_um=3., duplicate_cost=2.)


def validate_edges(coords, edges):
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    if not len(edges):
        return
    if edges.min() < 0 or edges.max() >= len(coords) or len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError('Invalid or repeated selected edge')
    if np.any(coords[edges[:, 1], 0] != coords[edges[:, 0], 0]+1):
        raise ValueError('Non-consecutive selected edge')
    if np.bincount(edges[:, 0], minlength=len(coords)).max() > 2 or np.bincount(edges[:, 1], minlength=len(coords)).max() > 1:
        raise ValueError('Selected graph violates lineage degrees')


def solve_window(coords, events, gains, node_gain, pinned, config):
    """Each event has [mother, daughter, optional second daughter or -1]."""
    n, e = len(coords), len(events)
    duplicates = []
    for t in np.unique(coords[:, 0]):
        ids = np.flatnonzero(coords[:, 0] == t)
        pairs = cKDTree(coords[ids, 1:]*SCALE).query_pairs(config['duplicate_radius_um'], output_type='ndarray')
        if len(pairs):
            duplicates.append(ids[pairs])
    duplicates = np.concatenate(duplicates) if duplicates else np.empty((0, 2), np.int64)
    # Variables: activation x, birth b, death d, events h, duplicate penalties u.
    size = 3*n+e+len(duplicates)
    cost = np.r_[-node_gain, np.full(n, config['birth_cost']), np.full(n, config['death_cost']),
                 -np.clip(gains, -12, 12), np.full(len(duplicates), config['duplicate_cost'])]
    rows, cols, values = [], [], []
    lower = np.r_[-pinned.astype(float), np.zeros(n), np.full(n, -np.inf), np.full(len(duplicates), -np.inf)]
    upper = np.r_[-pinned.astype(float), np.zeros(n), np.zeros(n), np.ones(len(duplicates))]
    def add(r, c, v):
        rows.extend(np.asarray(r).ravel().tolist()); cols.extend(np.asarray(c).ravel().tolist())
        values.extend(np.broadcast_to(v, np.asarray(r).shape).ravel().tolist())
    ids = np.arange(n)
    # incoming + birth - activation = -fixed incoming from prior window
    add(ids, ids, -1.); add(ids, n+ids, 1.)
    # one continuation OR one division OR death for each active mother
    add(n+ids, ids, -1.); add(n+ids, 2*n+ids, 1.)
    # A node needs an actual edge, not just a birth followed by immediate death.
    add(2*n+ids, n+ids, 1.); add(2*n+ids, 2*n+ids, 1.); add(2*n+ids, ids, -1.)
    event_ids = 3*n+np.arange(e)
    add(n+events[:, 0], event_ids, 1.)
    add(events[:, 1], event_ids, 1.)
    pair = events[:, 2] >= 0; add(events[pair, 2], event_ids[pair], 1.)
    r = 3*n+np.arange(len(duplicates))
    add(r, duplicates[:, 0], 1.); add(r, duplicates[:, 1], 1.); add(r, 3*n+e+np.arange(len(duplicates)), -1.)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), size)).tocsc()
    lb = np.zeros(size); ub = np.ones(size); lb[np.flatnonzero(pinned)] = 1.; ub[n+np.flatnonzero(pinned)] = 0.
    result = milp(cost, integrality=np.ones(size), bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options=dict(time_limit=config['time_limit'], mip_rel_gap=config['gap'], presolve=True))
    feasible = False
    if result.x is not None and np.isfinite(result.x).all():
        rounded = np.rint(result.x); lhs = matrix@rounded
        feasible = bool(np.all(rounded >= lb-1e-6) and np.all(rounded <= ub+1e-6) and
                        np.all(lhs >= lower-1e-6) and np.all(lhs <= upper+1e-6))
    if feasible:
        chosen = np.flatnonzero(rounded[3*n:3*n+e] > .5)
    else:
        # Explicitly recorded feasible fallback if HiGHS finds no incumbent.
        # It sacrifices look-ahead optimization, never degree constraints.
        used_mothers, used_daughters, chosen = set(), set(), []
        active = set(np.flatnonzero(pinned).tolist())
        duplicate_lookup = {}
        for a, b in duplicates:
            duplicate_lookup.setdefault(int(a), []).append(int(b)); duplicate_lookup.setdefault(int(b), []).append(int(a))
        for index in np.argsort(-gains, kind='stable'):
            event = events[index]; m = int(event[0]); daughters = [int(v) for v in event[1:] if v >= 0]
            if m in used_mothers or any(v in used_daughters for v in daughters):
                continue
            new = set([m, *daughters])-active
            penalty = sum(sum(v in active or (v in new and v < a) for v in duplicate_lookup.get(a, [])) for a in new)
            utility = gains[index]+sum(node_gain[list(new)])-config['duplicate_cost']*penalty
            utility -= config['birth_cost']*(m not in active)+config['death_cost']*len(daughters)
            if utility <= 0:
                continue
            chosen.append(int(index)); active.update(new); used_mothers.add(m); used_daughters.update(daughters)
        chosen = np.asarray(chosen, np.int64)
    gap = getattr(result, 'mip_gap', None)
    return chosen, dict(status=int(result.status), message=result.message, incumbent_feasible=feasible,
                         fallback=not feasible, gap=float(gap) if gap is not None and np.isfinite(gap) else None,
                         nodes=n, events=e, duplicate_pairs=len(duplicates), chosen=len(chosen))


def select_graph(graph, edges, edge_gains, triples, triple_gains, quality, config=None):
    config = dict(SOLVER_CONFIG if config is None else config)
    if not 0 < config['stride'] < config['window']:
        raise ValueError('Overlapping windows require 0 < stride < window')
    coords = graph['coords']; n = len(coords)
    events = np.concatenate((np.column_stack((edges, np.full(len(edges), -1))), triples)).astype(np.int64)
    gains = np.r_[edge_gains, triple_gains]
    if not np.isfinite(gains).all() or not np.isfinite(quality).all():
        raise ValueError('Invalid event or quality scores')
    for col in (1, 2):
        valid = events[:, col] >= 0
        if np.any(coords[events[valid, col], 0] != coords[events[valid, 0], 0]+1):
            raise ValueError('Event crosses non-consecutive frames')
    if np.any((events[:, 1] == events[:, 2]) & (events[:, 2] >= 0)):
        raise ValueError('Division daughters must differ')
    node_gain = .2*quality + np.where(graph['origin'] == 0, .1, -.15)
    chosen_global = []; reports = []; incoming = set()
    for first in range(0, int(graph['shape'][0])-1, config['stride']):
        end = min(first+config['window'], int(graph['shape'][0])-1)
        ids = np.flatnonzero((coords[:, 0] >= first) & (coords[:, 0] <= end))
        event_ids = np.flatnonzero((coords[events[:, 0], 0] >= first) & (coords[events[:, 0], 0] < end))
        if not len(ids) or not len(event_ids):
            incoming = set(); continue
        remap = np.full(n, -1, np.int64); remap[ids] = np.arange(len(ids))
        local = remap[events[event_ids].clip(0)]; local[events[event_ids] < 0] = -1
        pin = np.array([int(i) in incoming for i in ids])
        chosen, report = solve_window(coords[ids], local, gains[event_ids], node_gain[ids], pin, config)
        selected = event_ids[chosen]; cut = min(first+config['stride'], int(graph['shape'][0])-1)
        selected = selected[coords[events[selected, 0], 0] < cut]
        chosen_global.extend(selected.tolist())
        outgoing = events[selected, 1:].ravel(); outgoing = outgoing[outgoing >= 0]
        incoming = set(outgoing[coords[outgoing, 0] == cut].tolist())
        reports.append(dict(first=first, end=end, committed=len(selected), **report))
    selected_events = events[np.asarray(chosen_global, np.int64)]
    result = [(int(e[0]), int(d)) for e in selected_events for d in e[1:] if d >= 0]
    result = np.asarray(sorted(result), np.int64).reshape(-1, 2); validate_edges(coords, result)
    if len(np.unique(selected_events[:, 0])) != len(selected_events):
        raise ValueError('More than one outgoing event per mother')
    return result, dict(config=config, windows=reports, selected_events=len(selected_events),
                        selected_divisions=int((selected_events[:, 2] >= 0).sum()),
                        fallback_windows=sum(r['fallback'] for r in reports),
                        scope='Overlapping finite-horizon optimization with fixed committed incoming edges; not a global optimum certificate')
