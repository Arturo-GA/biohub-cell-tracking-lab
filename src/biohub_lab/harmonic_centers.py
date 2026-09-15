"""Extract Harmonic's image detector without its associations or graph reader."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from .detection_dag import DAGConfig, merge_sources
from .detector_proposals import SCALE


DETECTOR_CONFIG = dict(det_threshold=.965, det_tta=True, pool_kernel_um=3.,
    secondary_detection_weight=.80, minimum_candidate_retention=.90,
    edge_feature_tta=False, secondary_edge_feature_tta=False)
CHECKPOINTS = dict(primary='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771',
    secondary='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f')


def image_metadata(path, *, normalize=False, load_image=False, downsample=(1,4,4)):
    """Same strided shape and stored quantiles; never inspect a sibling GEFF."""
    import zarr
    path=Path(path)
    if path.suffix!='.zarr' or normalize or load_image:
        raise ValueError('Only image metadata access is allowed')
    group=zarr.open_group(str(path),mode='r');attrs=dict(group.attrs)
    scale=SCALE
    if 'multiscales' in attrs:
        transform=attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]
        if transform['type']!='scale':raise ValueError('Unexpected image transform')
        scale=np.asarray(transform['scale'][-3:])
    if not np.allclose(scale,SCALE,atol=1e-8,rtol=0):raise ValueError('Unexpected voxel scale')
    raw=group['0'].shape
    shape=raw[:1]+tuple((s+d-1)//d for s,d in zip(raw[1:],downsample))
    return SimpleNamespace(zarr_path=path,quantiles=attrs.get('image_statistics',{}).get('quantiles',{}),
        image_shape=shape,scale=tuple(scale))


def detector_source(predictor, trainer):
    """Keep the exact patched detector statements, remove the edge-prediction loop.

    The generated module has no upstream dataset reader, evaluation imports,
    graph builder, ILP entry point, or executable training routine.
    """
    parsed=ast.parse(predictor)
    wanted={'PredictConfig','load_model','_load_frame','pool_kernel_from_um','_detect_cells_pooled','predict_video'}
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    if {n.name for n in selected}!=wanted:raise ValueError('Detector function set changed')
    default=[n for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_DEFAULT_CONFIG' for t in n.targets)]
    model=[n for n in ast.parse(trainer).body if isinstance(n,ast.ClassDef) and n.name=='UNetNodeTransformer']
    if len(default)!=1 or len(model)!=1:raise ValueError('Model/config definitions changed')
    fn=next(n for n in selected if n.name=='predict_video')
    windows=[n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='ws']
    if len(windows)!=1:raise ValueError('Window loop changed')
    loop=windows[0]
    edge_loops=[n for n in loop.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(W - 1)']
    registries=[n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='coords_so_far' for t in n.targets)]
    if len(edge_loops)!=1 or len(registries)!=1:raise ValueError('Edge boundary changed')
    loop.body=[n for n in loop.body if n not in edge_loops+registries]
    # The old coordinate manifest is not needed; output receipts are written here.
    fn.body=[n for n in fn.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='_coordinate_manifest_arm')]
    forbidden={'predict_edges','extract_pos_features','augment_runtime','build_graph','save_graph','load_gt'}
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(fn)
        if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    if calls&forbidden:raise ValueError('Tracking or annotation call remains in detector')
    preamble='''from __future__ import annotations
import json, os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from tqdm import tqdm
from biohub_tracking.models import SimpleNodeTransformer, TemporalUNet3D
from biohub_lab.harmonic_centers import image_metadata as open_dataset
INTERACTIVE=False
_POS_EMBED_DIM=8
'''
    tree=ast.Module(body=ast.parse(preamble).body+model+default+selected,type_ignores=[])
    source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    compile(source,'<harmonic-centers>','exec')
    return source


def merge_primary(primary, auxiliary, shape, config=DAGConfig()):
    """Retain every primary center, then add novel E011 centers at the same radius.

    Scores rank only the two original auxiliary sources against each other.
    Primary centers have explicit precedence, not a fabricated confidence score.
    """
    primary=np.asarray(primary)
    if primary.ndim!=2 or primary.shape[1]!=4 or not np.isfinite(primary).all():
        raise ValueError('Invalid primary centers')
    if np.any(primary!=np.rint(primary)) or np.any(primary<0) or np.any(primary>np.asarray(shape)-1):
        raise ValueError('Primary centers must be native integer coordinates within the image')
    primary=primary.astype(np.int32)
    if len(np.unique(primary,axis=0))!=len(primary):raise ValueError('Duplicate primary centers')
    extra,scores,origin=merge_sources(auxiliary,shape,config)
    keep=np.ones(len(extra),bool)
    for t in np.unique(extra[:,0]):
        p=primary[primary[:,0]==t,1:];ids=np.flatnonzero(extra[:,0]==t)
        if len(p):keep[ids]=cKDTree(p*SCALE).query(extra[ids,1:]*SCALE)[0]>config.merge_um
    coords=np.concatenate([primary,extra[keep]])
    # Neutral primary score; source provenance, not cross-detector confidence.
    return coords,np.concatenate([np.ones(len(primary),np.float32),scores[keep]]),np.concatenate([
        np.zeros(len(primary),np.int8),origin[keep]])


def save_centers(path,coords,shape,metadata):
    path=Path(path);coords=np.asarray(coords,np.int32)
    # Validate without introducing a deduplication or temporal filter.
    empty=(np.empty((0,4),np.float32),np.empty(0,np.float32))
    merge_primary(coords,[empty],shape)
    np.savez_compressed(path,coords=coords,scores=np.ones(len(coords),np.float32),shape=np.asarray(shape,np.int32))
    receipt=dict(metadata,proposals=len(coords),frame_counts=[int((coords[:,0]==t).sum()) for t in range(shape[0])],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),detector_config=DETECTOR_CONFIG,
        coordinate_order='tzyx',stage='dual_seed_detector_before_any_association_or_track_filter',
        annotations_read=False,reference_graph_used=False)
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
