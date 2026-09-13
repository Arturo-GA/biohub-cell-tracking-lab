"""Image-derived 3D morphology + frozen HOCT + exact capacitated matching.

The feature order/statistics and parental normalization follow royerlab/hoct
2ccc5040823bc944ab67790abd1f56eea7cd4f05 (MIT; see licenses/HOCT.txt).
Our segmentation, batching and solver are separate implementations. This is
not the upstream two-pass gap/tracklet solver and requires no Gurobi license.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

MODEL_SHA256 = '5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33'
MEAN = np.array([463.26, 2.938, 356.49, 344.91, 11.521, .276, .966, .574,
                 .162, 167.81, -.027, .05, -.027, 87.012, -1.401, .05,
                 -1.401, 83.695, .009], dtype=np.float32)
STD = np.array([555.78, 7.6, 195.88, 226.10, 8.199, .216, .281, .193,
                .069, 678.45, 3.167, 2.875, 3.167, 512.92, 182.74, 2.875,
                182.74, 306.08, .078], dtype=np.float32)


@dataclass(frozen=True)
class HOCTConfig:
    scale: tuple = (1.625, .40625, .40625)
    mask_radius_um: float = 6.0
    mask_relative_threshold: float = .25
    max_distance_um: float = 14.0
    neighbors: int = 5
    window: int = 5
    window_stride: int = 3
    halo_um: float = 6.0
    max_context_edges: int = 1800
    appearance_cost: float = .5
    disappearance_cost: float = .25
    division_cost: float = .25
    edge_bias: float = .5


def segment_frame(image, points, config=HOCTConfig()):
    """Watershed instances constrained by measured intensity and physical radius.

    One marker per supplied detection, no extra nodes. The returned masks are
    approximate intensity basins, not learned FOCUS-3D segmentations.
    """
    from skimage.segmentation import watershed
    image = np.asarray(image, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Expected finite 3D image')
    low, high = float(image.min()), float(np.quantile(image, .999))
    normalized = (image - low) / max(high - low, 1e-7)
    if not len(points):
        return np.zeros(image.shape, np.int32), normalized
    indices = np.rint(points).astype(int)
    if (indices < 0).any() or (indices >= np.array(image.shape)).any():
        raise ValueError('Seed outside image')
    if len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError('Duplicate rounded seeds: refuse to silently merge cells')
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[tuple(indices.T)] = np.arange(1, len(points) + 1)
    smooth = ndi.gaussian_filter(normalized, sigma=(.45, 1., 1.))
    distance, nearest = ndi.distance_transform_edt(
        markers == 0, sampling=config.scale, return_indices=True)
    nearest_label = markers[tuple(nearest)]
    del nearest
    signal = np.r_[0., smooth[tuple(indices.T)]]
    cutoff = np.maximum(.02, config.mask_relative_threshold * signal[nearest_label])
    foreground = (distance <= config.mask_radius_um) & (smooth >= cutoff)
    foreground[tuple(indices.T)] = True
    labels = watershed(-smooth, markers=markers, mask=foreground).astype(np.int32)
    return labels, normalized


def frame_features(image, points, t, config=HOCTConfig()):
    """19 raw HOCT channels, retaining the detector centers for a fixed-node test."""
    from skimage.measure import regionprops
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    labels, normalized = segment_frame(image, points, config)
    features = np.zeros((len(points), 19), dtype=np.float32)
    features[:, 0] = t
    features[:, 1:4] = points
    sizes = np.zeros(len(points), dtype=np.int32)
    for region in regionprops(labels, intensity_image=normalized):
        i = region.label - 1
        features[i, 4:9] = (region.equivalent_diameter_area,
            region.intensity_min, region.intensity_max,
            region.intensity_mean, region.intensity_std)
        features[i, 9:18] = region.inertia_tensor.ravel()
        sizes[i] = region.area
    border = np.minimum(points, np.array(image.shape) - points).min(axis=1)
    features[:, 18] = 1. - np.minimum(1., border / 5.)
    if (sizes == 0).any() or not np.isfinite(features).all():
        raise ValueError('Missing mask or nonfinite morphology')
    return features, sizes


def candidate_edges(coords, config=HOCTConfig()):
    """Five candidate parents per target, physical cutoff, consecutive time only."""
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError('Expected finite t,z,y,x coordinates')
    if not np.equal(coords[:, 0], np.rint(coords[:, 0])).all():
        raise ValueError('Time must be integer')
    edges = []
    for t in np.unique(coords[:, 0]):
        source = np.flatnonzero(coords[:, 0] == t)
        target = np.flatnonzero(coords[:, 0] == t + 1)
        if not len(source) or not len(target):
            continue
        tree = cKDTree(coords[source, 1:] * config.scale)
        dist, index = tree.query(coords[target, 1:] * config.scale,
            k=list(range(1, min(config.neighbors, len(source)) + 1)),
            distance_upper_bound=config.max_distance_um)
        for j, target_id in enumerate(target):
            for d, i in zip(dist[j], index[j]):
                if np.isfinite(d):
                    edges.append((int(source[i]), int(target_id)))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _context_tiles(coords, edges, edge_ids, config):
    """Partition ownership by target; retain every parent alternative of a target.

    Halo context is auxiliary. Every edge is scored exactly once per window;
    spatial subdivision cannot silently drop a candidate or renormalize a
    target over only part of its parent set.
    """
    target_pos = coords[edges[edge_ids, 1], 1:] * config.scale
    initial_targets = np.unique(edges[edge_ids, 1])
    stack = [initial_targets]
    while stack:
        owned_targets = stack.pop()
        pos = coords[owned_targets, 1:] * config.scale
        low, high = pos.min(axis=0) - config.halo_um, pos.max(axis=0) + config.halo_um
        context = edge_ids[np.all((target_pos >= low) & (target_pos <= high), axis=1)]
        if len(context) > config.max_context_edges and len(owned_targets) > 1:
            axis = int(np.argmax(np.ptp(pos, axis=0)))
            ordered = owned_targets[np.argsort(pos[:, axis], kind='stable')]
            middle = len(ordered) // 2
            stack.extend([ordered[:middle], ordered[middle:]])
            continue
        if len(context) > config.max_context_edges:
            # Crowded halo: all alternatives for this target remain present.
            # Deterministically retain closest complete target groups as context.
            others = np.unique(edges[context, 1])
            distances = np.linalg.norm((coords[others, 1:] - pos[0] / config.scale) * config.scale, axis=1)
            groups, count = [], 0
            for target in others[np.argsort(distances, kind='stable')]:
                group = context[edges[context, 1] == target]
                if count + len(group) <= config.max_context_edges:
                    groups.append(group)
                    count += len(group)
            context = np.concatenate(groups)
        own = np.isin(edges[context, 1], owned_targets)
        expected = np.isin(edges[edge_ids, 1], owned_targets).sum()
        if own.sum() != expected:
            raise RuntimeError('Tile lost owned parent candidates')
        yield context, own, owned_targets


def load_hoct(path, device='cpu'):
    import torch
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('HOCT general_v1 checkpoint checksum mismatch')
    return torch.jit.load(str(path), map_location=device).eval()


def score_edges(model, coords, features, edges, config=HOCTConfig(), progress=None):
    """Frozen HOCT, overlapping 5-frame windows, median exp-logits, parent softmax."""
    import torch
    coords, features = np.asarray(coords), np.asarray(features)
    if features.shape != (len(coords), 19) or not np.isfinite(features).all():
        raise ValueError('Invalid HOCT features')
    if len(edges) == 0:
        return np.zeros(0), np.ones(len(coords)), {'batches': 0, 'max_edges': 0}
    device = next(model.parameters()).device
    standardized = (features - MEAN) / STD
    edge_samples = [[] for _ in edges]
    orphan_samples = [[] for _ in coords]
    t_min, t_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    starts = list(range(t_min, max(t_min + 1, t_max - config.window + 2), config.window_stride))
    last = max(t_min, t_max - config.window + 1)
    if starts[-1] != last:
        starts.append(last)
    batches, max_edges = 0, 0
    # FP32 is intentional: Kaggle T4 does not have native bfloat16 support.
    with torch.inference_mode():
        for start in starts:
            ids = np.flatnonzero((coords[edges[:, 0], 0] >= start)
                & (coords[edges[:, 1], 0] < start + config.window))
            if not len(ids):
                continue
            for context, own, owned_targets in _context_tiles(coords, edges, ids, config):
                nodes = np.unique(edges[context])
                local_edges = np.searchsorted(nodes, edges[context])
                positions = coords[nodes, 1:].astype(np.float32)
                edge_pos = positions[local_edges].mean(axis=1)
                def tensor(a):
                    return torch.as_tensor(a, device=device).unsqueeze(0)
                result = model(tensor(standardized[nodes]), tensor(positions), tensor(edge_pos),
                    tensor(local_edges), tensor(np.ones(len(nodes), bool)), tensor(np.ones(len(context), bool)))
                logits = result[0].float().cpu().numpy().reshape(-1)
                orphans = result[3].float().cpu().numpy().reshape(-1)
                if logits.shape != (len(context),) or not np.isfinite(logits).all() or not np.isfinite(orphans).all():
                    raise RuntimeError('Invalid model output')
                for e, value in zip(context[own], np.exp(np.minimum(logits[own], 20.))):
                    edge_samples[e].append(float(value))
                for target in owned_targets:
                    i = np.searchsorted(nodes, target)
                    orphan_samples[target].append(float(np.exp(min(orphans[i], 20.))))
                batches += 1
                max_edges = max(max_edges, len(context))
            if progress:
                progress({'window_start': start, 'batches': batches})
    if any(not values for values in edge_samples):
        raise RuntimeError('Temporal windows left candidate edges unscored')
    edge_exp = np.array([np.median(v) for v in edge_samples])
    orphan_exp = np.array([np.median(v) if v else 1. for v in orphan_samples])
    sums = np.bincount(edges[:, 1], weights=edge_exp, minlength=len(coords))
    denominator = np.maximum(sums + orphan_exp, 1e-30)
    return edge_exp / denominator[edges[:, 1]], orphan_exp / denominator, {
        'batches': batches, 'max_edges': max_edges,
        'min_edge_observations': min(map(len, edge_samples))}


def solve_lineage(coords, edges, probabilities, orphan, config=HOCTConfig()):
    """Exact fixed-node, dt=1 HOCT objective via two slots per source.

    Slot 1 refunds disappearance, slot 2 incurs division. Each target chooses
    one source slot or its own appearance dummy. Because slot 1 is cheaper,
    slot 2 cannot be chosen alone at optimum. Matches the additive single-pass
    ILP objective when every detection is retained and gaps are excluded.
    """
    coords, edges = np.asarray(coords), np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    probabilities, orphan = np.asarray(probabilities), np.asarray(orphan)
    if probabilities.shape != (len(edges),) or orphan.shape != (len(coords),):
        raise ValueError('Invalid probability shape')
    if not np.isfinite(probabilities).all() or not np.isfinite(orphan).all():
        raise ValueError('Nonfinite probability')
    if ((probabilities < 0) | (probabilities > 1)).any() or ((orphan < 0) | (orphan > 1)).any():
        raise ValueError('Probability out of range')
    if len(edges) and (np.any(coords[edges[:, 1], 0] != coords[edges[:, 0], 0] + 1)
                      or len(np.unique(edges, axis=0)) != len(edges)):
        raise ValueError('Expected unique consecutive edges')
    if config.disappearance_cost < 0 or config.division_cost < 0:
        raise ValueError('Slot ordering requires nonnegative costs')
    selected = []
    for t in np.unique(coords[:, 0]):
        source = np.flatnonzero(coords[:, 0] == t)
        target = np.flatnonzero(coords[:, 0] == t + 1)
        if not len(target) or not len(source):
            continue
        take = np.flatnonzero(coords[edges[:, 0], 0] == t)
        if not len(take):
            continue
        pair = edges[take]
        rows = np.searchsorted(target, pair[:, 1])
        cols = np.searchsorted(source, pair[:, 0])
        base = config.edge_bias - probabilities[take] - config.appearance_cost * (1 - orphan[pair[:, 1]])
        costs = np.r_[base - config.disappearance_cost, base + config.division_cost,
                      np.zeros(len(target))]
        # Sparse matching treats zeros as absent, so shift ALL assignments equally.
        shift = 1. + max(0., -float(costs.min()))
        matrix = coo_matrix((costs + shift,
            (np.r_[rows, rows, np.arange(len(target))],
             np.r_[cols, cols + len(source), 2 * len(source) + np.arange(len(target))])),
            shape=(len(target), 2 * len(source) + len(target))).tocsr()
        r, c = min_weight_full_bipartite_matching(matrix)
        selected.extend((int(source[j % len(source)]), int(target[i]))
                        for i, j in zip(r, c) if j < 2 * len(source))
    return np.asarray(sorted(selected), dtype=np.int64).reshape(-1, 2)
