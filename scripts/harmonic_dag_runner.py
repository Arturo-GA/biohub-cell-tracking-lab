"""E012: add primary Harmonic detections to E011, then audit three frozen arms."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from notebook_runner import competition_dir
from biohub_lab.detection_dag import DAGConfig,build_dag,save_dag,select_development
from biohub_lab.dag_coverage import CoverageGate,evaluate_graph,readiness
from biohub_lab.harmonic_centers import CHECKPOINTS,DETECTOR_CONFIG,detector_source,merge_primary,save_centers


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def verify_cached(package,input_root=Path('/kaggle/input')):
    spec=read(Path(package)/'baseline/harmonic_dag_inputs.json')
    roots=[input_root/spec['slug'],input_root/'notebooks/jarturo'/spec['slug']]
    matches=[p/'detection_dag_experiment' for p in roots if (p/'detection_dag_experiment/frozen_inputs.json').is_file()]
    if len(matches)!=1:raise ValueError('Expected one E011 input')
    cached=matches[0]
    if sha(cached/'frozen_inputs.json')!=spec['frozen_sha256']:raise ValueError('E011 version changed')
    # This manifest was written before E011 annotation access. Do not open its result/coverage files.
    frozen=read(cached/'frozen_inputs.json')
    if frozen['annotations_read'] or not frozen['all_detection_graphs_complete']:raise ValueError('Invalid frozen provenance')
    if frozen['datasets']!=spec['datasets']:raise ValueError('Cached datasets changed')
    for name,files in spec['files'].items():
        for suffix,digest in files.items():
            path=cached/'videos'/name/suffix
            if sha(path)!=digest:raise ValueError('E011 cached checksum changed: '+name+'/'+suffix)
    return cached,spec


def generate_video(name,coords,shape,root,cached,detector_metadata):
    root=Path(root);folder=root/'videos'/name;folder.mkdir(parents=True,exist_ok=True)
    primary_receipt=save_centers(folder/'harmonic.npz',coords,shape,detector_metadata)
    auxiliary=[];proposal_files={}
    for method in ('cellect','gaussian'):
        path=Path(cached)/'videos'/name/(method+'.npz');data=arrays(path)
        if not np.array_equal(data['shape'],shape):raise ValueError('Cached image shape changed')
        auxiliary.append((data['coords'],data['scores']));proposal_files[method]=sha(path)
    combined=merge_primary(coords,auxiliary,shape)
    arms={'harmonic':(np.asarray(coords,np.int32),np.ones(len(coords),np.float32),np.zeros(len(coords),np.int8)),
          'combined':combined}
    receipts={}
    for arm,(points,scores,origin) in arms.items():
        graph=build_dag(points,scores,origin,shape)
        receipts[arm]=save_dag(graph,folder/arm,name)
    if not np.array_equal(combined[0][:len(coords)],coords):raise ValueError('Primary centers were not preserved')
    result=dict(dataset=name,shape=list(map(int,shape)),primary=primary_receipt,
        cached_proposal_sha256=proposal_files,arms=receipts,annotations_read=False,reference_graph_used=False,
        primary_centers_preserved=len(coords),added_auxiliary_centers=len(combined[0])-len(coords))
    write(folder/'generation.json',result)
    return result


def worker(data,root,names,cached,runtime,index):
    import torch
    import zarr
    torch.set_num_threads(2)
    root=Path(root);runtime=read(runtime)
    spec=importlib.util.spec_from_file_location('harmonic_image_detector',runtime['module'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    device=torch.device('cuda')
    primary,window,downsample=module.load_model(Path(runtime['primary']),device)
    secondary,other_window,other_downsample=module.load_model(Path(runtime['secondary']),device)
    if window!=other_window or downsample!=other_downsample:raise ValueError('Model grid mismatch')
    if window!=2 or tuple(downsample)!=(1,4,4):raise ValueError('Pinned detector grid changed')
    cfg=module.PredictConfig(det_threshold=DETECTOR_CONFIG['det_threshold'],det_tta=True,pool_kernel_um=3.)
    progress=dict(worker=index,completed=[],status='generating',annotations_read=False)
    receipt_path=root/f'worker_{index}.json';write(receipt_path,progress)
    try:
        for name in names:
            start=time.monotonic();progress['current_video']=name;write(receipt_path,progress)
            image=Path(data)/(name+'.zarr');shape=zarr.open_group(str(image),mode='r')['0'].shape
            coords,edges=module.predict_video(primary,image,device,cfg,window_size=window,
                downsample=downsample,secondary_model=secondary,secondary_detection_weight=.80)
            if edges:raise ValueError('Image-only detector unexpectedly returned associations')
            metadata=dict(checkpoint_sha256=runtime['checkpoint_sha256'],
                detector_source_sha256=runtime['detector_source_sha256'],window_size=window,downsample=list(downsample))
            generated=generate_video(name,coords,shape,root,cached,metadata)
            progress['completed'].append(name);write(receipt_path,progress)
            print('PRIMARY_DAG_READY',name,'primary',len(coords),'added',generated['added_auxiliary_centers'],
                'seconds',time.monotonic()-start,flush=True)
        progress.update(status='complete');progress.pop('current_video',None);write(receipt_path,progress)
    except Exception as error:
        progress.update(status='failed',error=str(error));write(receipt_path,progress);raise


def run_workers(data,root,names,cached,runtime,repo):
    import torch
    count=min(2,torch.cuda.device_count())
    if not count:raise RuntimeError('E012 requires a GPU runtime')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    tokens=[s.strip() for s in visible.split(',')] if visible and visible!='-1' else list(map(str,range(count)))
    if len(tokens)<count:raise ValueError('CUDA visibility mismatch')
    processes=[]
    try:
        for index in range(count):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=tokens[index],BIOHUB_GPU_SHARD=str(index),
                BIOHUB_EDGE_FEATURE_TTA='0',BIOHUB_SECONDARY_EDGE_FEATURE_TTA='0',BIOHUB_DIAGNOSTIC_ARM='',
                BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION='.90',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
            env['PYTHONPATH']=str(Path(repo)/'src')+os.pathsep+env['PYTHONPATH']
            command=[sys.executable,'-u',__file__,'worker',str(data),str(root),json.dumps(names[index::count]),str(cached),str(runtime),str(index)]
            processes.append(subprocess.Popen(command,env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Primary detector worker failed')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Primary detector worker failed')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate();process.wait()


def freeze_graphs(root,names,cached,pins):
    records={}
    for name in names:
        folder=Path(root)/'videos'/name;generation=read(folder/'generation.json')
        if generation['annotations_read'] or generation['reference_graph_used']:raise ValueError('Invalid generation provenance')
        files=[folder/'harmonic.npz',folder/'harmonic.json',folder/'generation.json']
        for arm in ('harmonic','combined'):
            if generation['arms'][arm]['config']!=asdict(DAGConfig()):raise ValueError('DAG settings changed')
            if sha(folder/arm/'graph.npz')!=generation['arms'][arm]['graph_sha256']:raise ValueError('Generated graph changed')
            files.extend(folder/arm/filename for filename in ('graph.npz','trajectory_samples.npz','receipt.json'))
        if sha(folder/'harmonic.npz')!=generation['primary']['sha256']:raise ValueError('Primary detections changed')
        for suffix,digest in pins['files'][name].items():
            if sha(Path(cached)/'videos'/name/suffix)!=digest:raise ValueError('Cached graph/proposals changed')
        records[name]=dict(generation=generation,files={p.relative_to(root).as_posix():sha(p) for p in files},
            baseline_graph_sha256=pins['files'][name]['dag/graph.npz'])
    frozen=dict(datasets=names,records=records,annotations_read=False,all_detection_graphs_complete=True)
    write(Path(root)/'frozen_inputs.json',frozen)
    return frozen


def audit_frozen(root,data,cached,frozen):
    from biohub_lab.temporal_data import load_gt
    root=Path(root);statistics={arm:{} for arm in ('e011','harmonic','combined')}
    for name in frozen['datasets']:
        paths={'e011':Path(cached)/'videos'/name/'dag/graph.npz',
            **{arm:root/'videos'/name/arm/'graph.npz' for arm in ('harmonic','combined')}}
        record=frozen['records'][name]
        hashes={'e011':record['baseline_graph_sha256'],**{arm:record['generation']['arms'][arm]['graph_sha256'] for arm in ('harmonic','combined')}}
        if any(sha(p)!=hashes[arm] for arm,p in paths.items()):raise ValueError('Graph changed before annotation access')
        truth,edges=load_gt(Path(data)/(name+'.geff'))
        for arm,path in paths.items():
            counts,witnesses=evaluate_graph(arrays(path),truth,edges,match_um=CoverageGate().match_um)
            statistics[arm][name]=counts
            folder=root/'coverage'/arm;folder.mkdir(parents=True,exist_ok=True)
            write(folder/(name+'.json'),dict(counts=counts,annotation_conditioned_witnesses=witnesses,
                scope='Coverage evidence only, never predictions or submission'))
            if sha(path)!=hashes[arm]:raise ValueError('Graph changed during annotation audit')
        print('PAIRED_COVERAGE_READY',name,json.dumps({arm:v[name] for arm,v in statistics.items()}),flush=True)
    result={arm:dict(readiness(rows),per_video=rows) for arm,rows in statistics.items()}
    write(root/'readiness.json',result)
    return result


def main(package,repo,primary,secondary,root=Path('/kaggle/working/harmonic_dag_experiment'),input_root=Path('/kaggle/input')):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);package=Path(package);repo=Path(repo);start=time.monotonic()
    receipt=dict(experiment='E012',status='verifying_inputs',config=asdict(DAGConfig()),gate=asdict(CoverageGate()),
        detector_config=DETECTOR_CONFIG,neural_training=False,leaderboard_submitted=False,
        annotations_used_for_generation=False,harmonic_associations_used=False,
        scope='Paired coverage audit on the same 48 development videos as E011. Same two acquisition groups; '
            'public secondary detector saw these videos during training. Not independent validation or official score.')
    path=root/'result.json';write(path,receipt)
    try:
        cached,pins=verify_cached(package,input_root)
        manifest=read(package/'baseline/detection_dag_dev.json')
        if sha(package/'baseline/detection_dag_dev.json')!=pins['development_manifest_sha256']:raise ValueError('Development manifest changed')
        if manifest['config']!=receipt['config'] or manifest['gate']!=receipt['gate']:raise ValueError('Audit settings changed')
        comp=competition_dir();data=comp/'train';names=sorted(p.stem for p in data.glob('*.zarr'))
        if hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()!=manifest['training_names_sha256']:raise ValueError('Train inventory changed')
        selected=select_development(names,manifest['excluded'],manifest['per_group'])
        if selected!=manifest['development'] or selected!=pins['datasets']:raise ValueError('Cohort changed')
        if set(selected)&{p.stem for p in (comp/'test').glob('*.zarr')}:raise ValueError('Test/development overlap')
        weights={'primary':str(primary),'secondary':str(secondary)}
        for key,value in weights.items():
            if sha(value)!=CHECKPOINTS[key]:raise ValueError('Public checkpoint changed')
            if not Path(value).with_name('config.json').is_file():raise ValueError('Missing detector model configuration')
        module=root/'image_detector.py'
        predictor=repo/'scripts/predict_unet_transformer.py';trainer=repo/'scripts/train_unet_transformer.py'
        module.write_text(detector_source(predictor.read_text(),trainer.read_text()))
        runtime=dict(weights,module=str(module),checkpoint_sha256=CHECKPOINTS,detector_source_sha256=sha(module),
            patched_predictor_sha256=sha(predictor),trainer_sha256=sha(trainer),
            config_sha256={key:sha(Path(p).with_name('config.json')) for key,p in weights.items()})
        runtime_path=root/'runtime.json';write(runtime_path,runtime)
        receipt.update(status='generating_primary_detections_and_two_dags',datasets=selected,
            development_manifest_sha256=sha(package/'baseline/detection_dag_dev.json'),cached_input_manifest_sha256=sha(package/'baseline/harmonic_dag_inputs.json'),
            runtime=runtime,source_hashes={str(p.relative_to(package)):sha(p) for p in (
                package/'src/biohub_lab/detection_dag.py',package/'src/biohub_lab/dag_coverage.py',
                package/'src/biohub_lab/harmonic_centers.py',package/'scripts/harmonic_dag_runner.py')})
        write(path,receipt);run_workers(data,root,selected,cached,runtime_path,repo)
        frozen=freeze_graphs(root,selected,cached,pins)
        receipt.update(status='evaluating_frozen_coverage',generation_seconds=time.monotonic()-start,frozen_inputs_sha256=sha(root/'frozen_inputs.json'))
        write(path,receipt);evaluated=audit_frozen(root,data,cached,frozen)
        if sha(root/'frozen_inputs.json')!=receipt['frozen_inputs_sha256']:raise ValueError('Frozen manifest changed')
        receipt.update(status='complete',seconds=time.monotonic()-start,arms=evaluated,
            delta_vs_e011={key:evaluated['combined']['totals'][key]-evaluated['e011']['totals'][key] for key in evaluated['e011']['totals']},
            totals={arm:{key:sum(r['generation']['arms'][arm][key] for r in frozen['records'].values()) for key in (
                'nodes','mother_daughter_edges','continuation_edges','division_pairs','sample_queries','sample_queries_with_paths')} for arm in ('harmonic','combined')},
            decision=evaluated['combined']['decision'])
        write(path,receipt);print('HARMONIC_DAG_RESULT',json.dumps(receipt,indent=2),flush=True)
        return receipt
    except Exception as error:
        receipt.update(failed_stage=receipt['status'],status='failed',error=str(error),seconds=time.monotonic()-start)
        write(path,receipt);(root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),sys.argv[5],sys.argv[6],int(sys.argv[7]))
