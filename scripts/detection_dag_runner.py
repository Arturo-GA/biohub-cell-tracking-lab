"""E011: fresh raw-image detections, frozen path DAGs, separate development audit."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

from biohub_lab.detection_dag import DAGConfig,select_development,merge_sources,build_dag,save_dag
from biohub_lab.dag_coverage import CoverageGate,evaluate_graph,readiness
from notebook_runner import competition_dir


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,record):Path(path).write_text(json.dumps(record,indent=2)+'\n')


def generate_video(data,root,name,weights):
    from biohub_lab.cellect_detector import run_video as cellect
    from biohub_lab.gaussian_detector import run_video as gaussian
    start=time.monotonic();data=Path(data);root=Path(root)
    folder=root/'videos'/name;folder.mkdir(parents=True,exist_ok=True)
    sources=[];receipts={};shape=None
    for method,detector in [('cellect',cellect),('gaussian',gaussian)]:
        path=folder/(method+'.npz')
        result=detector(data/(name+'.zarr'),path,weights=weights) if method=='cellect' else detector(data/(name+'.zarr'),path)
        if sha(path)!=result['sha256']:raise ValueError('Proposal save mismatch')
        with np.load(path,allow_pickle=False) as values:
            if shape is not None and tuple(values['shape'])!=shape:raise ValueError('Detector shape disagreement')
            shape=tuple(map(int,values['shape']));sources.append((values['coords'],values['scores']))
        receipts[method]=dict(sha256=result['sha256'],count=result['proposals'])
    coords,scores,origin=merge_sources(sources,shape)
    graph=build_dag(coords,scores,origin,shape)
    result=save_dag(graph,folder/'dag',name)
    result.update(dataset=name,shape=shape,proposal_files=receipts,seconds=time.monotonic()-start)
    write(folder/'generation.json',result)
    return result


def worker(data,root,names,weights,index):
    import torch
    torch.set_num_threads(2)
    progress=dict(worker=index,completed=[],status='generating',annotations_read=False)
    path=Path(root)/f'worker_{index}.json';write(path,progress)
    try:
        for name in names:
            progress['current_video']=name;write(path,progress)
            result=generate_video(data,root,name,weights)
            progress['completed'].append(name);write(path,progress)
            print('DETECTION_DAG_READY',name,json.dumps(result),flush=True)
        progress.update(status='complete');progress.pop('current_video',None);write(path,progress)
    except Exception as error:
        progress.update(status='failed',error=str(error));write(path,progress);raise


def run_workers(data,root,names,weights):
    import torch
    count=min(2,torch.cuda.device_count())
    if not count:raise RuntimeError('E011 detector audit requires GPU runtime')
    processes=[]
    try:
        for index in range(count):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
            command=[sys.executable,'-u',__file__,'worker',str(data),str(root),json.dumps(names[index::count]),str(weights),str(index)]
            processes.append(subprocess.Popen(command,env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Detection/DAG worker failed; inspect worker receipts')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Detection/DAG worker failed')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate();process.wait()


def freeze_graphs(root,names):
    records={}
    for name in names:
        folder=Path(root)/'videos'/name
        generation=json.loads((folder/'generation.json').read_text())
        if generation['annotations_read'] or generation['reference_graph_used']:raise ValueError('Invalid inference provenance')
        if generation['config']!=asdict(DAGConfig()):raise ValueError('DAG configuration changed')
        files=[folder/'dag/graph.npz',folder/'dag/trajectory_samples.npz',folder/'generation.json',
               folder/'cellect.npz',folder/'gaussian.npz']
        if sha(folder/'dag/graph.npz')!=generation['graph_sha256']:raise ValueError('Graph digest changed')
        for method in ('cellect','gaussian'):
            if sha(folder/(method+'.npz'))!=generation['proposal_files'][method]['sha256']:raise ValueError('Proposal digest changed')
        records[name]=dict(generation=generation,files={str(p.relative_to(root)):sha(p) for p in files})
    freeze=dict(datasets=names,records=records,annotations_read=False,all_detection_graphs_complete=True)
    write(Path(root)/'frozen_inputs.json',freeze)
    return freeze


def audit_frozen(root,data,freeze):
    # GEFF access is confined to this stage, after ALL development DAGs exist.
    from biohub_lab.temporal_data import load_gt
    root=Path(root);data=Path(data);output=root/'coverage';output.mkdir(exist_ok=True)
    statistics={}
    for name in freeze['datasets']:
        path=root/'videos'/name/'dag/graph.npz'
        if sha(path)!=freeze['records'][name]['generation']['graph_sha256']:raise ValueError('Graph changed before coverage audit')
        with np.load(path,allow_pickle=False) as values:graph={key:values[key] for key in values.files}
        truth,edges=load_gt(data/(name+'.geff'))
        counts,details=evaluate_graph(graph,truth,edges,match_um=CoverageGate().match_um)
        write(output/(name+'.json'),dict(counts=counts,annotation_conditioned_witnesses=details,
            scope='Post-freeze coverage witnesses only; never use as model predictions or submission'))
        if sha(path)!=freeze['records'][name]['generation']['graph_sha256']:raise ValueError('Graph changed during coverage audit')
        statistics[name]=counts
        print('COVERAGE_READY',name,json.dumps(counts),flush=True)
    result=readiness(statistics);result['per_video']=statistics
    write(root/'readiness.json',result)
    return result


def main(package,root=Path('/kaggle/working/detection_dag_experiment')):
    from biohub_lab.cellect_detector import find_checkpoint
    root=Path(root);root.mkdir(parents=True,exist_ok=True);package=Path(package);start=time.monotonic()
    receipt=dict(experiment='E011',status='verifying_development_selection',config=asdict(DAGConfig()),
        gate=asdict(CoverageGate()),scope='48 development videos separate from the four previous diagnostic videos. '
            'Same two acquisition groups and videos used in earlier model training; not an unseen end-to-end holdout. '
            'CELLECT training membership is unverified. No model is trained or selected here.',
        annotations_used_for_generation=False,harmonic_graph_used=False,neural_training=False,
        leaderboard_submitted=False,local_monitor_requested=False)
    result_path=root/'result.json';write(result_path,receipt)
    try:
        manifest_path=package/'baseline/detection_dag_dev.json';manifest=json.loads(manifest_path.read_text())
        if manifest['config']!=receipt['config'] or manifest['gate']!=receipt['gate']:raise ValueError('Frozen settings differ')
        comp=competition_dir();data=comp/'train';names=sorted(p.stem for p in data.glob('*.zarr'))
        inventory_sha=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()
        if inventory_sha!=manifest['training_names_sha256']:raise ValueError('Training inventory changed')
        selected=select_development(names,manifest['excluded'],manifest['per_group'])
        if selected!=manifest['development']:raise ValueError('Development selection changed')
        if set(selected)&{p.stem for p in (comp/'test').glob('*.zarr')}:raise ValueError('Test video in development set')
        weights=find_checkpoint('/kaggle/input')
        receipt.update(status='generating_detections_and_dags',datasets=selected,
            development_manifest_sha256=sha(manifest_path),source_hashes={str(p.relative_to(package)):sha(p) for p in (
                package/'src/biohub_lab/detection_dag.py',package/'src/biohub_lab/dag_coverage.py',
                package/'scripts/detection_dag_runner.py')})
        write(result_path,receipt)
        run_workers(data,root,selected,weights)
        frozen=freeze_graphs(root,selected)
        receipt.update(status='evaluating_frozen_coverage',generation_seconds=time.monotonic()-start,
            frozen_inputs_sha256=sha(root/'frozen_inputs.json'))
        write(result_path,receipt)
        result=audit_frozen(root,data,frozen)
        if sha(root/'frozen_inputs.json')!=receipt['frozen_inputs_sha256']:raise ValueError('Frozen manifest changed')
        receipt.update(status='complete',seconds=time.monotonic()-start,readiness=result,
            totals={key:sum(record['generation'][key] for record in frozen['records'].values()) for key in (
                'nodes','mother_daughter_edges','continuation_edges','division_pairs','sample_queries','sample_queries_with_paths')},
            decision=result['decision'])
        write(result_path,receipt);print('DETECTION_DAG_RESULT',json.dumps(receipt,indent=2),flush=True)
        return receipt
    except Exception as error:
        receipt.update(failed_stage=receipt['status'],status='failed',error=str(error),seconds=time.monotonic()-start)
        write(result_path,receipt);(root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),sys.argv[5],int(sys.argv[6]))
    else:main(sys.argv[1])
