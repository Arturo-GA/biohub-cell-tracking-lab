"""Prepare one image-only video locally, reusing verified detector/feature stages."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import zarr
from biohub_lab.event_data import extract_visual, neighborhoods, save_json
from biohub_lab.event_portable import sha, signature
from biohub_lab.event_train import load_arrays
from biohub_lab.harmonic_centers import CHECKPOINTS, DETECTOR_CONFIG, merge_primary, save_centers
from biohub_lab.detection_dag import build_dag, save_dag
from event_stages import device_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','video','image','primary','secondary','cellect','module','support-repo'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cached-e012');parser.add_argument('--gaussian-cache')
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    device_policy(args.device);torch.set_num_threads(2);started=time.monotonic()
    split=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']
    if args.video not in split['fit']+split['calibration']+split['evaluation']:
        raise ValueError('Video outside fixed E013 partition')
    image=Path(args.image)
    if image.name != args.video+'.zarr':raise ValueError('Image/video identity mismatch')
    folder=Path(args.root)/'videos'/args.video;folder.mkdir(parents=True,exist_ok=True)
    for key in ('primary','secondary'):
        if sha(getattr(args,key)) != CHECKPOINTS[key]:raise ValueError('Public checkpoint changed')
        config=Path(getattr(args,key)).with_name('config.json')
        expected=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['config_sha256']
        if sha(config) != expected:raise ValueError('Public model configuration changed')
    module_hash=hashlib.sha256(Path(args.module).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if module_hash != '53d4569a57d95edfe35f2605d38ed26a46c7b5ddebbe25523410a138b003f070':
        raise ValueError('Use the pinned E012 image-only detector module')
    image_files={p.relative_to(image).as_posix():sha(p) for p in sorted(image.rglob('*')) if p.is_file()}
    model_sources={p.relative_to(Path(args.support_repo)/'src').as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                   for p in sorted((Path(args.support_repo)/'src/biohub_tracking/models').rglob('*.py'))}
    identity=dict(image_sha256=signature(image_files),checkpoints=CHECKPOINTS,module_sha256=module_hash,
                  model_sources=model_sources,cellect_sha256=sha(args.cellect),detector=DETECTOR_CONFIG,
                  feature_mode='single_view_first_seen_dual_UNet_32_plus_32_trilinear',device_type=torch.device(args.device).type)
    receipt_path=folder/'generation.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('portable_identity')==identity and all(sha(folder/f)==d for f,d in receipt['files'].items()):
            print('GENERATION_ALREADY_COMPLETE',args.video);return
        raise ValueError('Existing generation differs; preserve/version it')
    old_identity=folder/'preparation_identity.json'
    if old_identity.exists() and json.loads(old_identity.read_text()) != identity:
        raise ValueError('Partial preparation inputs changed')
    save_json(old_identity,identity)
    sys.path.insert(0,str(Path(args.support_repo)/'src'))
    os.environ.update(BIOHUB_EDGE_FEATURE_TTA='0',BIOHUB_SECONDARY_EDGE_FEATURE_TTA='0',BIOHUB_DIAGNOSTIC_ARM='',BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION='.90')
    spec=importlib.util.spec_from_file_location('portable_image_detector',args.module)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    device=torch.device(args.device);shape=zarr.open_group(str(image),mode='r')['0'].shape
    def stage_complete(name):
        path=folder/name;pin=folder/(name+'.sha256')
        if path.exists() and pin.exists():
            if sha(path)!=pin.read_text().strip():raise ValueError('Partial stage changed: '+name)
            return True
        return False
    def pin(name):
        (folder/(name+'.sha256')).write_text(sha(folder/name)+'\n')
    if args.gaussian_cache and not stage_complete('gaussian.npz'):
        cache=Path(args.gaussian_cache);record=json.loads((cache/'gaussian_receipt.json').read_text())
        from biohub_lab.gaussian_detector import GAUSSIAN_CONFIG
        if (record['video']!=args.video or record['image_sha256']!=identity['image_sha256'] or
            record['config']!=GAUSSIAN_CONFIG or record['annotations_read'] or
            sha(cache/'gaussian.npz')!=record['proposals_sha256']):
            raise ValueError('CPU Gaussian package provenance mismatch')
        shutil.copyfile(cache/'gaussian.npz',folder/'gaussian.npz');pin('gaussian.npz')
    cached=(Path(args.cached_e012)/'videos'/args.video/'combined/graph.npz') if args.cached_e012 else None
    if not stage_complete('graph.npz'):
        if cached and cached.is_file():
            pins=json.loads((ROOT/'baseline/event_graph_inputs.json').read_text())
            if sha(cached)!=pins['files'][args.video]['combined/graph.npz']:raise ValueError('E012 graph changed')
            shutil.copyfile(cached,folder/'graph.npz')
        else:
            if args.video in split['evaluation']:raise ValueError('Evaluation must reuse pinned E012 graph')
            from biohub_lab.cellect_detector import run_video as cellect
            from biohub_lab.gaussian_detector import run_video as gaussian
            for name,runner in [('cellect.npz',cellect),('gaussian.npz',gaussian)]:
                if not stage_complete(name):
                    kwargs=dict(weights=args.cellect,device=args.device) if name=='cellect.npz' else {}
                    runner(image,folder/name,**kwargs);pin(name)
                    if device.type=='cuda':torch.cuda.empty_cache()
            if not stage_complete('harmonic.npz'):
                primary,window,ds=module.load_model(Path(args.primary),device)
                secondary,w2,ds2=module.load_model(Path(args.secondary),device)
                if window!=2 or w2!=2 or tuple(ds)!=(1,4,4) or tuple(ds2)!=tuple(ds):raise ValueError('Frozen architecture changed')
                cfg=module.PredictConfig(det_threshold=.965,det_tta=True,pool_kernel_um=3.)
                coords,associations=module.predict_video(primary,image,device,cfg,window_size=window,downsample=ds,
                    secondary_model=secondary,secondary_detection_weight=.80)
                if associations:raise ValueError('Image-only detector returned associations')
                save_centers(folder/'harmonic.npz',coords,shape,dict(checkpoint_sha256=CHECKPOINTS));pin('harmonic.npz')
                del primary,secondary
                if device.type=='cuda':torch.cuda.empty_cache()
            base=load_arrays(folder/'harmonic.npz');aux=[load_arrays(folder/name) for name in ('cellect.npz','gaussian.npz')]
            points,scores,origin=merge_primary(base['coords'],[(a['coords'],a['scores']) for a in aux],shape)
            save_dag(build_dag(points,scores,origin,shape),folder,args.video)
        pin('graph.npz')
    graph=load_arrays(folder/'graph.npz')
    if not np.array_equal(graph['shape'],shape):raise ValueError('Image shape differs from graph')
    for key in ('primary','secondary'):
        filename='visual_'+key+'.npy'
        if not stage_complete(filename):
            model,window,ds=module.load_model(Path(getattr(args,key)),device)
            if window!=2 or tuple(ds)!=(1,4,4):raise ValueError('Frozen feature architecture changed')
            visual=extract_visual(module,[model],image,graph['coords'],device)
            np.save(folder/filename,visual,allow_pickle=False);pin(filename)
            del model,visual
            if device.type=='cuda':torch.cuda.empty_cache()
    visual=np.concatenate([np.load(folder/('visual_'+key+'.npy'),allow_pickle=False) for key in ('primary','secondary')],axis=1)
    np.savez_compressed(folder/'features.npz',visual=visual,neighbors=neighborhoods(graph['coords']))
    receipt=dict(dataset=args.video,annotations_read=False,harmonic_associations_used=False,portable_identity=identity,
        nodes=len(graph['coords']),shape=list(map(int,shape)),seconds=time.monotonic()-started,
        files={f:sha(folder/f) for f in ('graph.npz','features.npz')})
    save_json(receipt_path,receipt);print('PORTABLE_VIDEO_READY',args.video,json.dumps(receipt),flush=True)


if __name__=='__main__':main()
