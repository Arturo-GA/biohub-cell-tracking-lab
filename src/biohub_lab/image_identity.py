"""E030: real encoder tokens, physical image views and joint lineage scoring.

No annotations enter extract_tokens(), ImageIdentityModel or decode_events().
Sparse supervision is constructed separately by the data preparation runner.
"""
from itertools import combinations
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch.nn import functional as F

SCALE = np.array([1.625, .40625, .40625])


def physical_views(volume, center, size=64, spacing=.40625):
    """XY/XZ/YZ sections with equal physical extent; trilinear, border padded."""
    center = np.asarray(center, float)
    if volume.ndim != 3 or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Expected finite ZYX center and 3D image')
    offset = (np.arange(size) - (size-1)/2) * spacing
    yy, xx = np.meshgrid(offset, offset, indexing='ij')
    views = []
    for a, b in ((1, 2), (0, 2), (0, 1)):
        grid = np.broadcast_to(center[:, None, None], (3, size, size)).copy()
        grid[a] += yy/SCALE[a]; grid[b] += xx/SCALE[b]
        views.append(map_coordinates(volume, grid, order=1, mode='nearest'))
    return np.asarray(views, np.float32)


def extract_tokens(net, views):
    """Capture normalized patch tokens BEFORE Cellpose's segmentation readout.

    CPDINO.forward returns random compatibility styles; those are never read.
    The hook is at net.out, whose input excludes CLS/storage tokens already.
    A central 2x2 patch pool and whole-crop pool retain nucleus/context evidence.
    """
    if net.training:
        raise ValueError('Encoder must be in eval mode')
    batch, channels, height, width = views.shape
    if channels != 1 or height != width or height % net.ps:
        raise ValueError('Expected square single-channel views divisible by stride')
    captured = []
    def capture(module, args):
        captured.append(args[0])
    hook = net.out.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            net(views)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError('Expected one normalized token tensor')
    tokens = captured[0]; side = height//net.ps
    if tokens.ndim != 3 or tokens.shape[:2] != (batch, side*side):
        raise ValueError('Unexpected CPDINO token geometry')
    grid = tokens.reshape(batch, side, side, -1)
    mid = side//2
    center = grid[:, mid-1:mid+1, mid-1:mid+1].mean((1, 2))
    context = tokens.mean(1)
    result = torch.cat((F.normalize(center, dim=-1), F.normalize(context, dim=-1)), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite encoder descriptors')
    return result


class ImageIdentityModel(nn.Module):
    """Shared appearance embedding; parent competition; symmetric division head."""
    def __init__(self, features=4608, hidden=96, use_image=True):
        super().__init__(); self.use_image = use_image
        self.embedding = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge = nn.Sequential(nn.Linear(2*hidden+5, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.identity = nn.Sequential(nn.Linear(2*hidden+4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.division = nn.Sequential(nn.Linear(3*hidden+6, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null = nn.Parameter(torch.zeros(()))

    def encode(self, features):
        if not self.use_image:
            features = torch.zeros_like(features)
        return F.normalize(self.embedding(features), dim=-1)

    def edge_logits(self, h, coords, edges):
        a, b = edges.T; delta = (coords[b, 1:]-coords[a, 1:]) * coords.new_tensor(SCALE)/10
        geo = torch.cat((delta, delta.norm(dim=-1, keepdim=True), (coords[b, :1]-coords[a, :1])), -1)
        return self.edge(torch.cat(((h[a]-h[b]).abs(), h[a]*h[b], geo), -1)).squeeze(-1)

    def identity_logits(self, h, coords, pairs):
        a, b = pairs.T; delta = ((coords[b, 1:]-coords[a, 1:])*coords.new_tensor(SCALE)/10).abs()
        geo = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
        return self.identity(torch.cat(((h[a]-h[b]).abs(), h[a]*h[b], geo), -1)).squeeze(-1)

    def division_logits(self, h, coords, triples):
        m, a, b = triples.T
        da = (coords[a, 1:]-coords[m, 1:])*coords.new_tensor(SCALE)/10
        db = (coords[b, 1:]-coords[m, 1:])*coords.new_tensor(SCALE)/10
        geo = torch.cat(((da+db)/2, (da-db).abs()), -1)
        appearance = torch.cat((h[m], (h[a]+h[b])/2, (h[a]-h[b]).abs()), -1)
        return self.division(torch.cat((appearance, geo), -1)).squeeze(-1)


def candidate_events(coords, k=8, radius=20.):
    """Geometry only proposal generation; both parent and daughter competition."""
    from scipy.spatial import cKDTree
    coords = np.asarray(coords); edges = []
    for t in np.unique(coords[:, 0]):
        a = np.flatnonzero(coords[:, 0] == t); b = np.flatnonzero(coords[:, 0] == t+1)
        if not len(a) or not len(b): continue
        d, ix = cKDTree(coords[a, 1:]*SCALE).query(coords[b, 1:]*SCALE, k=k, distance_upper_bound=radius)
        d, ix = d.reshape(len(b), k), ix.reshape(len(b), k)
        row, col = np.nonzero(np.isfinite(d))
        edges.extend(zip(a[ix[row, col]].tolist(), b[row].tolist()))
    edges = np.asarray(sorted(set(edges)), np.int64).reshape(-1, 2)
    triples = []
    for mother in np.unique(edges[:, 0]):
        daughters = edges[edges[:, 0] == mother, 1]
        order = np.argsort(np.linalg.norm((coords[daughters, 1:]-coords[mother, 1:])*SCALE, axis=1), kind='stable')[:k]
        triples.extend((mother, a, b) for a, b in combinations(sorted(daughters[order].tolist()), 2))
    return edges, np.asarray(triples, np.int64).reshape(-1, 3)


def decode_events(coords, edges, edge_gains, triples, division_gains, duplicates=(), time_limit=15.):
    """Joint node/event MILP with learned duplicate exclusivity and binary forks.

    Each selected mother has one continuation OR one two-daughter event. Nodes
    have at most one incoming event. No unlabeled data or GT used. Scores are
    explicit gains supplied by the caller; negative events need not be chosen.
    """
    coords = np.asarray(coords); n = len(coords)
    edges = np.asarray(edges, np.int64).reshape(-1, 2); triples = np.asarray(triples, np.int64).reshape(-1, 3)
    events = np.concatenate((np.c_[edges, np.full(len(edges), -1)], triples))
    gains = np.r_[edge_gains, division_gains]; e = len(events)
    if not np.isfinite(gains).all() or len(gains) != e:
        raise ValueError('Invalid gains')
    if not e: return np.empty((0, 2), np.int64), dict(status='empty', selected_events=0)
    for m, a, b in events:
        if m < 0 or a < 0 or max(m, a, b) >= n or a == b or b < -1:
            raise ValueError('Invalid event index')
        if any(coords[d, 0] != coords[m, 0]+1 for d in (a, b) if d >= 0):
            raise ValueError('Events must connect consecutive frames')
    rr, cc, vv, upper = [], [], [], []
    def row(entries, bound):
        r = len(upper); upper.append(bound)
        for c, v in entries: rr.append(r); cc.append(c); vv.append(v)
    for node in range(n):
        incoming = np.flatnonzero((events[:, 1] == node) | (events[:, 2] == node))
        outgoing = np.flatnonzero(events[:, 0] == node)
        row([(int(i), 1.) for i in incoming]+[(e+node, -1.)], 0.)
        row([(int(i), 1.) for i in outgoing]+[(e+node, -1.)], 0.)
        row([(e+node, 1.)]+[(int(i), -1.) for i in np.union1d(incoming, outgoing)], 0.)
    for a, b in duplicates:
        if not (0 <= a < n and 0 <= b < n) or a == b or coords[a, 0] != coords[b, 0]:
            raise ValueError('Duplicate constraints require distinct same-frame nodes')
        row([(e+int(a), 1.), (e+int(b), 1.)], 1.)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(upper), e+n)).tocsc()
    result = milp(np.r_[-gains, np.full(n, 1e-7)], integrality=np.ones(e+n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, -np.inf, upper), options=dict(time_limit=time_limit, mip_rel_gap=.001))
    if result.x is None: raise RuntimeError('No feasible graph incumbent')
    x = np.rint(result.x)
    if np.any(x < 0) or np.any(x > 1) or np.any(matrix@x > np.asarray(upper)+1e-6):
        raise RuntimeError('Invalid rounded graph incumbent')
    chosen = events[x[:e] > .5]
    selected = [(int(m), int(d)) for m, a, b in chosen for d in (a, b) if d >= 0]
    return np.asarray(sorted(selected), np.int64).reshape(-1, 2), dict(status=int(result.status), selected_events=len(chosen), divisions=int((chosen[:, 2] >= 0).sum()), optimal=result.status == 0)


def infer_lineage(model, features, coords, *, division_logit_offset, identity_logit_threshold, k=8, radius=20.):
    """Detector-independent inference adapter; operating points MUST be supplied.

    Accepts a mixed-source set of detections with descriptors from their images.
    Offsets require development calibration on actual detections, not the E030
    annotation-centered diagnostic. This API does not authorize a submission.
    """
    from scipy.spatial import cKDTree
    if not np.isfinite([division_logit_offset,identity_logit_threshold]).all():
        raise ValueError('Finite explicit operating points required')
    coords=np.asarray(coords,np.float32);features=np.asarray(features,np.float32)
    if coords.ndim!=2 or coords.shape[1]!=4 or len(features)!=len(coords) or not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError('Invalid detection coordinates/descriptors')
    edges,triples=candidate_events(coords,k,radius)
    same=[]
    for t in np.unique(coords[:,0]):
        ids=np.flatnonzero(coords[:,0]==t)
        pairs=cKDTree(coords[ids,1:]*SCALE).query_pairs(7.,output_type='ndarray')
        same.extend(ids[pairs].tolist())
    same=np.asarray(same,np.int64).reshape(-1,2)
    device=next(model.parameters()).device;model.eval()
    with torch.inference_mode():
        c=torch.as_tensor(coords,device=device);h=model.encode(torch.as_tensor(features,device=device))
        es=model.edge_logits(h,c,torch.as_tensor(edges,device=device))-model.null
        ts=model.division_logits(h,c,torch.as_tensor(triples,device=device))-division_logit_offset
        # A division is one alternative event with two child parent gains.
        if len(triples):
            for col in (1,2):
                ts=ts+model.edge_logits(h,c,torch.as_tensor(triples[:,[0,col]],device=device))-model.null
        ds=model.identity_logits(h,c,torch.as_tensor(same,device=device))
    duplicates=same[ds.cpu().numpy()>=identity_logit_threshold]
    selected,report=decode_events(coords,edges,es.cpu().numpy(),triples,ts.cpu().numpy(),duplicates)
    report.update(candidate_edges=len(edges),candidate_divisions=len(triples),duplicate_conflicts=len(duplicates))
    return selected,report
