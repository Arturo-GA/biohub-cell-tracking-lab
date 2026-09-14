"""Label-free native-coordinate proposals and the Harmonic runtime adapter."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625])
CONFIG = dict(dedup_um=3., temporal_um=3., max_addition_fraction=.15,
              min_addition_budget=16, coordinate_order='tzyx', seed=20260914)


def nms(coords, scores, distance_um=3., scale=SCALE):
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores)
    if len(coords) != len(scores) or not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid proposals')
    if not len(coords):
        return np.empty(0, np.int64)
    tree = cKDTree(coords * scale)
    removed = np.zeros(len(coords), bool)
    keep = []
    for i in np.argsort(-scores, kind='stable'):
        if removed[i]:
            continue
        keep.append(i)
        removed[tree.query_ball_point(coords[i] * scale, distance_um)] = True
    return np.asarray(keep, np.int64)


def temporal_filter(frames, support=None, distance_um=3.):
    """At least one adjacent frame supports each proposal; boundary uses one side."""
    support = [p for p, _ in frames] if support is None else support
    trees = [cKDTree(np.asarray(p).reshape(-1, 3) * SCALE) for p in support]
    result = []
    for t, (points, scores) in enumerate(frames):
        keep = np.zeros(len(points), bool)
        for other in (t - 1, t + 1):
            if 0 <= other < len(trees) and len(points):
                keep |= trees[other].query(points * SCALE)[0] <= distance_um
        result.append((points[keep], scores[keep]))
    return result


def save_proposals(path, frames, shape, metadata):
    coords, scores = [], []
    for t, (p, s) in enumerate(frames):
        p = np.asarray(p, np.float32).reshape(-1, 3)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > np.asarray(shape[1:]) - 1):
            raise ValueError('Out-of-volume proposals')
        coords.append(np.column_stack([np.full(len(p), t), p]))
        scores.append(s)
    coords = np.concatenate(coords).astype(np.float32)
    scores = np.concatenate(scores).astype(np.float32)
    path = Path(path)
    np.savez_compressed(path, coords=coords, scores=scores, shape=np.array(shape))
    receipt = dict(metadata, proposals=len(coords), frame_counts=[len(p) for p, _ in frames],
                   config=CONFIG, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   inputs='raw .zarr images only; no GEFF, ground truth, or cached prediction inputs')
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


@lru_cache(maxsize=8)
def _load(path):
    with np.load(path, allow_pickle=False) as data:
        coords, scores, shape = data['coords'], data['scores'], data['shape']
    if coords.shape != (len(scores), 4) or not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid proposal file')
    if np.any(coords < 0) or np.any(coords > shape - 1) or np.any(coords[:, 0] != coords[:, 0].astype(int)):
        raise ValueError('Invalid proposal coordinates')
    return coords, scores, shape


def augment_detections(arr, proposals, scores, t, downsample, shape):
    """Keep runtime detections first, then novel centers; return downsampled grid."""
    arr = np.asarray(arr).reshape(-1, 4)
    proposals = np.asarray(proposals).reshape(-1, 3)
    downsample = np.asarray(downsample)
    keep = nms(proposals, scores, CONFIG['dedup_um'])
    if len(arr) and len(keep):
        distances = cKDTree(arr[:, 1:] * downsample * SCALE).query(proposals[keep] * SCALE)[0]
        keep = keep[distances > CONFIG['dedup_um']]
    budget = max(CONFIG['min_addition_budget'], int(np.ceil(len(arr) * CONFIG['max_addition_fraction'])))
    keep = keep[:budget]
    # Integer native positions prevent truncation from creating untracked subvoxel differences.
    native = np.clip(np.rint(proposals[keep]), 0, np.array(shape[1:]) - 1)
    extra = np.column_stack([np.full(len(keep), t), native / downsample])
    return np.concatenate([arr, extra]).astype(np.float32)


def augment_runtime(arr, ds_path, t, downsample):
    root = Path(os.environ['BIOHUB_PROPOSAL_DIR'])
    coords, scores, shape = _load(str(root / (Path(ds_path).stem + '.npz')))
    mask = coords[:, 0] == t
    result = augment_detections(arr, coords[mask, 1:], scores[mask], t, downsample, shape)
    record = dict(dataset=Path(ds_path).stem, t=int(t), baseline=len(arr),
                  proposals=int(mask.sum()), added=len(result) - len(arr))
    shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
    with (root / f'injection_{shard}.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    return result


def patch_predictor(source):
    anchor = '                coord_offset[t] = (global_node_count, global_node_count + len(arr))'
    if source.count(anchor) != 1 or 'augment_runtime' in source:
        raise ValueError('Detection registry anchor changed or already patched')
    result = source.replace(anchor,
        '                from biohub_lab.detector_proposals import augment_runtime\n'
        '                arr = augment_runtime(arr, ds_path, t, downsample)\n' + anchor, 1)
    compile(result, '<augmented-predictor>', 'exec')
    return result


def install_predictor_file(predictor_path, receipt_path):
    """Install with this module's imports, independent of the caller's globals."""
    predictor_path, receipt_path = Path(predictor_path), Path(receipt_path)
    original = predictor_path.read_text(encoding='utf8')
    changed = patch_predictor(original)
    receipt = {
        'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'after_sha256': hashlib.sha256(changed.encode()).hexdigest(),
        'stage': 'after baseline integrity verification and TTA patches, before detection/graph/ILP',
    }
    predictor_path.write_text(changed, encoding='utf8')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    return receipt


def install_pipeline_hook(source):
    anchor = 'print("secondary edge-feature TTA patch installed and enabled", flush=True)'
    if source.count(anchor) != 1:
        raise ValueError('Final runtime patch anchor changed')
    return source.replace(anchor, anchor + '''
from biohub_lab.detector_proposals import install_predictor_file
install_predictor_file(_ps, WORKING_DIR / "proposal_patch_receipt.json")
print("Native-resolution proposals installed before coordinate registration", flush=True)
''', 1)
