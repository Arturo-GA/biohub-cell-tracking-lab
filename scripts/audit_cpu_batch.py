"""Verify a recovered image-only CPU batch and, optionally, its local features."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import zarr
from biohub_lab.event_portable import sha,signature
from biohub_lab.event_data import neighborhoods
from biohub_lab.event_train import load_arrays
from biohub_lab.gaussian_detector import GAUSSIAN_CONFIG
from biohub_lab.harmonic_centers import CHECKPOINTS,DETECTOR_CONFIG
from import_cpu_inputs import import_package
from run_local_prepared_batch import packages


def read(path):return json.loads(Path(path).read_text())


def audit_inputs(folder,launch_path):
    folder=Path(folder);launch=read(launch_path);state=read(folder/'status.json')
    downloaded=read(folder/'download_verification.json')
    batch=read(folder/'local_inputs/batch_result.json')
    assert state['status']=='COMPLETE' and state['kernel']==launch['kernel']
    assert downloaded['payload_sha256']==launch['payload_sha256']
    assert launch['is_private'] and not launch['enable_gpu'] and not launch['enable_tpu']
    split_sha=hashlib.sha256((ROOT/'baseline/event_graph_split.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    rows=[]
    for source in packages(folder/'local_inputs'):
        result=read(source/'result.json');manifest=read(source/'image_package.json')
        gaussian=read(source/'gaussian_receipt.json');name=result['video']
        assert manifest['video']==gaussian['video']==name and manifest['offset']==result['offset']
        assert manifest['split_manifest_sha256']==split_sha
        assert not result['annotations_read'] and not manifest['annotations_read'] and not gaussian['annotations_read']
        assert gaussian['config']==GAUSSIAN_CONFIG
        assert signature(manifest['image_files'])==manifest['image_sha256']==gaussian['image_sha256']
        imported=import_package(source,ROOT/'outputs/local_images')
        shape=list(zarr.open_group(imported['image'],mode='r')['0'].shape)
        assert shape==manifest['shape']
        assert sha(source/manifest['archive'])==result['archive_sha256']
        assert sha(source/'gaussian.npz')==result['gaussian_sha256']==gaussian['proposals_sha256']
        with np.load(source/'gaussian.npz',allow_pickle=False) as proposals:
            coords=proposals['coords'];count=len(coords)
            assert coords.shape==(count,4) and np.isfinite(coords).all()
            assert np.array_equal(proposals['shape'],shape)
            assert (coords>=0).all() and (coords<=np.array(shape)-1).all()
        rows.append(dict(video=name,offset=result['offset'],shape=shape,source_bytes=manifest['source_bytes'],
            image_files=imported['verified_files'],image_sha256=manifest['image_sha256'],
            archive_sha256=result['archive_sha256'],gaussian_sha256=result['gaussian_sha256'],
            gaussian_proposals=count,seconds=result['seconds']))
        print('CPU_CHILD_VERIFIED',name,flush=True)
    assert sum(row['source_bytes'] for row in rows)==batch['source_bytes']
    return dict(kernel=launch['kernel'],version=launch['version'],status='COMPLETE',
        checked_at_utc=state['checked_at_utc'],verified_at_utc=datetime.now(timezone.utc).isoformat(),
        offset=batch['offset'],next_offset=batch['next_offset'],videos=rows,
        source_bytes=batch['source_bytes'],payload_sha256=launch['payload_sha256'],
        verification=dict(remote_sources_match_launch_payload=True,exact_contiguous_cohort=True,
            all_archives_and_imported_image_files=True,split_manifest=True,gaussian_hashes_config_and_bounds=True),
        annotations_read=False,gpu=False,tpu=False,training_started=False)


def audit_features(cpu_receipt,root):
    root=Path(root);rows=[]
    for item in cpu_receipt['videos']:
        name=item['video'];folder=root/'videos'/name;receipt=read(folder/'generation.json')
        identity=receipt['portable_identity']
        assert not receipt['annotations_read'] and not receipt['harmonic_associations_used']
        assert identity['image_sha256']==item['image_sha256'] and identity['device_type']=='cuda'
        assert identity['checkpoints']==CHECKPOINTS and identity['detector']==DETECTOR_CONFIG
        assert all(sha(folder/f)==digest for f,digest in receipt['files'].items())
        graph=load_arrays(folder/'graph.npz');features=load_arrays(folder/'features.npz')
        count=receipt['nodes'];coords=graph['coords']
        assert coords.shape==(count,4) and np.isfinite(coords).all()
        assert np.array_equal(graph['shape'],item['shape'])
        assert (coords>=0).all() and (coords<=graph['shape']-1).all()
        assert features['visual'].shape==(count,64) and features['visual'].dtype==np.float16
        assert np.isfinite(features['visual']).all()
        assert np.array_equal(features['neighbors'],neighborhoods(coords))
        assert np.array_equal(features['visual'],np.concatenate([
            np.load(folder/('visual_'+key+'.npy'),allow_pickle=False) for key in ('primary','secondary')],axis=1))
        assert sha(folder/'gaussian.npz')==item['gaussian_sha256']
        stages={}
        for key in ('gaussian','cellect','harmonic'):
            file=folder/(key+'.npz');assert sha(file)==(folder/(key+'.npz.sha256')).read_text().strip()
            stages[key]=len(load_arrays(file)['coords'])
        assert np.array_equal(coords[:stages['harmonic']],load_arrays(folder/'harmonic.npz')['coords'])
        guard=[json.loads(line) for line in (folder/'retention_guard_single.jsonl').read_text().splitlines()]
        # A stage retry may leave previous log entries; require every frame in the completed suffix.
        assert len(guard)>=item['shape'][0]
        latest=guard[-item['shape'][0]:]
        assert sorted(row['frame'] for row in latest)==list(range(item['shape'][0]))
        assert all(row['dataset']==name and row['minimum_retention']==.9 for row in latest)
        rows.append(dict(video=name,nodes=count,shape=item['shape'],stages=stages,files=receipt['files'],
            seconds=receipt['seconds'],torch_peak_allocated_bytes=receipt['torch_peak_allocated_bytes'],
            torch_peak_reserved_bytes=receipt['torch_peak_reserved_bytes'],process_peak_rss_bytes=receipt['process_peak_rss_bytes'],
            measurement_scope=receipt['measurement_scope'],reused_stages=receipt['reused_stages']))
        print('LOCAL_CHILD_VERIFIED',name,count,flush=True)
    return dict(status='complete',verified_at_utc=datetime.now(timezone.utc).isoformat(),videos=rows,
        verification=dict(generation_and_stage_hashes=True,image_identity=True,feature_shape_dtype_and_finiteness=True,
            sequential_feature_concatenation_exact=True,complete_neighborhoods_recomputed=True,
            graph_coordinate_bounds=True,primary_centers_preserved=True,retention_log_complete=True),
        annotations_read=False,training_started=False,
        scope='Real detector and feature preparation on the fixed input partition; not a candidate score.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--launch',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--local-output')
    args=parser.parse_args();result=audit_inputs(args.folder,args.launch)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if args.local_output:
        local=audit_features(result,ROOT/'outputs/local_event_graph')
        Path(args.local_output).write_text(json.dumps(local,indent=2)+'\n')
    print('BATCH_AUDIT_PASSED',len(result['videos']),flush=True)
