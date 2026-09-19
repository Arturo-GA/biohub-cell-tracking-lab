"""E014: endpoint appearance and geometry with within-mother ranking.

Only E013 known labels participate. Unannotated candidates stay unknown.
The two heads have independent encoders to avoid cross-task interference.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALE = np.array([1.625, .40625, .40625], np.float32)


def node_features(graph, visual):
    if visual.shape != (len(graph['coords']), 64) or not np.isfinite(visual).all():
        raise ValueError('Invalid cached appearance')
    # Absolute time/location are omitted: they cannot identify annotated paths.
    return np.concatenate((visual.astype(np.float32),
        np.eye(3, dtype=np.float32)[graph['origin']],
        np.tanh(graph['scores'][:, None]).astype(np.float32)), axis=1)


class AssociationRanker(nn.Module):
    def __init__(self, division=False, width=64):
        super().__init__()
        self.config = dict(division=division, width=width)
        self.division = division
        self.appearance = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, width), nn.SiLU())
        geometry = 10 if division else 4
        metadata = 12 if division else 8
        self.head = nn.Sequential(nn.Linear(4*width+geometry+metadata, 128), nn.SiLU(),
            nn.LayerNorm(128), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, nodes, positions):
        h = self.appearance(nodes[..., :64])
        if not self.division:
            a, b = h.unbind(1)
            delta = positions[:, 1]-positions[:, 0]
            geometry = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
            features = torch.cat((a, b, (a-b).abs(), a*b, geometry,
                                  nodes[..., 64:].flatten(1)), -1)
        else:
            m, a, b = h.unbind(1)
            da = positions[:, 1]-positions[:, 0]; db = positions[:, 2]-positions[:, 0]
            lengths = torch.stack((da.norm(dim=-1), db.norm(dim=-1)), -1).sort(-1).values
            geometry = torch.cat(((da+db)/2, (da-db).abs(), lengths,
                (da-db).norm(dim=-1, keepdim=True), (da*db).sum(-1, keepdim=True)), -1)
            ma, mb = nodes[:, 1, 64:], nodes[:, 2, 64:]
            metadata = torch.cat((nodes[:, 0, 64:], (ma+mb)/2, (ma-mb).abs()), -1)
            features = torch.cat((m, (a+b)/2, (a-b).abs(), a*b, geometry, metadata), -1)
        return self.head(features).squeeze(-1)


def competing_groups(indices, y):
    """Return row IDs for mothers with both known positive and negative events."""
    order = np.argsort(indices[:, 0], kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(indices[order, 0]))+1, len(order)]
    groups = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        ids = order[start:end]; p = ids[y[ids] == 1]; n = ids[y[ids] == 0]
        if len(p) and len(n):
            groups.append((p, n))
    return groups


def sample_rows(y, groups, rng, batch=512, ranking_pairs=128):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise ValueError('Balanced training requires both known classes')
    rows = np.r_[rng.choice(pos, batch//2), rng.choice(neg, batch-batch//2)]
    pairs = []
    if groups:
        for index in rng.integers(len(groups), size=ranking_pairs):
            p, n = groups[index]; pairs.append((rng.choice(p), rng.choice(n)))
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    return rows, pairs


def ranking_loss(logits, labels, pair_logits):
    value = F.binary_cross_entropy_with_logits(logits, labels)
    if pair_logits.numel():
        value = value + F.softplus(pair_logits[:, 1]-pair_logits[:, 0]).mean()
    return value


def average_precision(scores, y):
    """Threshold-grouped AP; ties cannot gain credit from input ordering."""
    scores = np.asarray(scores); y = np.asarray(y)
    if not np.isfinite(scores).all() or not np.isin(y, (0, 1)).all():
        raise ValueError('Invalid ranking metric input')
    if not y.sum() or y.sum() == len(y):
        return None
    order = np.argsort(-scores, kind='stable'); sorted_scores = scores[order]
    tp = np.cumsum(y[order]); ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order)-1]
    precision = tp[ends]/(ends+1); recall = tp[ends]/y.sum()
    return float(np.sum(np.diff(np.r_[0., recall])*precision))


def evaluate_ranking(scores, labels, groups):
    ap = average_precision(scores, labels)
    # Mothers, not counts of duplicate positive correspondences, have equal weight.
    correct = [float(scores[p].max() > scores[n].max()) for p, n in groups]
    return dict(average_precision=ap, positive_fraction=float(np.mean(labels)),
        competing_mothers=len(groups), top_positive_fraction=float(np.mean(correct)) if correct else None,
        examples=len(labels), positives=int(labels.sum()))
