"""Local motion evidence adjusts positions without adding/deleting detections.

ctx_graph is supplied by the shared context_rules module. All gates are fixed
before evaluation, distances are micrometres and forks are protected.
"""
import numpy as np


def e070_positions(raw, edges, smoothed, cfg):
    out = {k: dict(v) for k, v in smoothed.items()}
    changed = 0
    if 'fork_smooth' in cfg:
        out, changed = topology_smoothing(raw, edges, out, cfg['fork_smooth'])
    mode = cfg.get('position_mode')
    if mode is None:
        return out, changed
    assert mode in ('curvature', 'outlier')
    pos, pred, succ = ctx_graph(raw, edges)
    scale = np.array([1.625, .40625, .40625])
    for s in sorted(raw):
        if len(pred[s]) != 1 or len(succ[s]) != 1:
            continue
        p, d = pred[s][0], succ[s][0]
        if len(succ[p]) != 1 or len(pred[p]) != 1 or len(pred[d]) != 1 or len(succ[d]) != 1:
            continue
        pp, q = pred[p][0], succ[d][0]
        if len(succ[pp]) != 1 or len(pred[q]) != 1:
            continue
        if [int(raw[k]['t']) for k in (pp, p, s, d, q)] != list(range(int(raw[s]['t'])-2, int(raw[s]['t'])+3)):
            continue
        if mode == 'curvature':
            acceleration = np.array([pos[s]-2*pos[p]+pos[pp], pos[d]-2*pos[s]+pos[p], pos[q]-2*pos[d]+pos[s]])
            norms = np.linalg.norm(acceleration, axis=1)
            if np.min(norms) < .5:
                continue
            direction = acceleration / norms[:, None]
            if np.dot(direction[0], direction[1]) < .8 or np.dot(direction[1], direction[2]) < .8:
                continue
            # Retain more of a coherent curved path instead of fitting a line.
            new = pos[s] + (.35/.8) * (np.array([out[s][a] for a in ('z','y','x')])*scale-pos[s])
        else:
            forward, backward = 2*pos[p]-pos[pp], 2*pos[d]-pos[q]
            expected = (forward+backward)/2
            residual = float(np.linalg.norm(pos[s]-expected))
            if np.linalg.norm(forward-backward) > 1. or not 2. < residual < 6.:
                continue
            # Both sides agree on a local outlier; limit correction to 2 um.
            shift = expected-pos[s]
            new = pos[s] + min(.5, 2./residual)*shift
        for axis, value in zip(('z','y','x'), new/scale):
            out[s][axis] = float(value)
        changed += 1
    return out, changed
