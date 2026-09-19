"""CPU-only calibration of structured decoding, then one frozen 48-video evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from biohub_lab.association_rank import AssociationRanker
from biohub_lab.association_cpu import score_events
from biohub_lab.event_solver import select_graph
from biohub_lab.evaluate import evaluate_csv
from biohub_lab.submission import COLUMNS


def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def arrays(p):
    with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}


def locate(input_root,relative):
    found=[]
    # Kernel outputs can be mounted under flat slugs or notebooks/user/slug.
    for parent in [input_root,*input_root.glob('*'),*input_root.glob('*/*'),*input_root.glob('*/*/*')]:
        p=parent/relative
        if p.is_file():found.append(p)
    found=list(dict.fromkeys(found))
    if len(found)!=1:raise ValueError('Expected exactly one attached file: '+relative)
    return found[0]


def write_predictions_csv(folder,names,target):
    with target.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(COLUMNS);index=0
        for name in names:
            d=arrays(folder/name/'prediction.npz');nodes=np.unique(d['edges'])
            if not len(nodes):raise ValueError('Empty selected graph: '+name)
            for node in nodes:
                writer.writerow([index,name,'node',int(node),*map(int,d['coords'][node]),-1,-1]);index+=1
            for a,b in d['edges']:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(a),int(b)]);index+=1


def evaluation_dir(train,names,dest):
    dest.mkdir(parents=True,exist_ok=True)
    for name in names:
        for suffix in ('.zarr','.geff'):
            source=train/(name+suffix);target=dest/source.name
            if not source.exists():raise FileNotFoundError(source)
            if not target.exists():target.symlink_to(source,target_is_directory=True)
    return dest


def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/association_cpu')):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if torch.cuda.is_available():raise RuntimeError('Use a CPU session for this workflow')
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(status='verifying_inputs',accelerator='none',training_started=False,leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        pins=read(package/'baseline/e014_cpu_pins.json');split=read(package/'baseline/event_graph_split.json')['split']
        e013=locate(input_root,'event_graph_experiment/frozen_inputs.json').parent
        e014=locate(input_root,'association_rank_experiment/result.json').parent
        if sha(e013/'frozen_inputs.json')!=pins['frozen_inputs_sha256'] or sha(e014/'result.json')!=pins['e014_result_sha256']:
            raise ValueError('Attached source run changed')
        frozen=read(e013/'frozen_inputs.json')['records'];models={}
        for head in ('edge','division'):
            path=e014/f'best_{head}.pt'
            if sha(path)!=pins['checkpoint_sha256'][head]:raise ValueError('Checkpoint changed')
            state=torch.load(path,map_location='cpu',weights_only=False)
            model=AssociationRanker(**state['model_config']);model.load_state_dict(state['state_dict']);model.eval();models[head]=model
        train=None
        for comp in (input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'):
            if (comp/'train').is_dir():train=comp/'train';break
        if train is None:raise FileNotFoundError('Competition data missing')
        thresholds=pins['thresholds'];save(root/'frozen_configuration.json',pins)
        # Compare only on calibration: links alone vs learned divisions.
        # No threshold grid or repeated evaluation-cohort selection.
        selected_arm=None
        for role in ('calibration','evaluation'):
            names=split[role];arms=['continuations','divisions'] if role=='calibration' else [selected_arm]
            result.update(status='predicting_'+role,current_video=None);save(root/'result.json',result)
            for name in names:
                started=time.monotonic();source=e013/'videos'/name
                for filename,digest in frozen[name]['files'].items():
                    if sha(source/filename)!=digest:raise ValueError('Graph or features changed: '+name)
                graph=arrays(source/'graph.npz')
                with np.load(source/'features.npz',allow_pickle=False) as features:
                    scored,stats=score_events(graph,features['visual'],models,thresholds)
                for arm in arms:
                    values=dict(scored)
                    if arm=='continuations':values.update(triples=np.empty((0,3),np.int64),triple_gains=np.empty(0,np.float32))
                    edges,solver=select_graph(graph,**values)
                    dest=root/role/arm/name;dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/'prediction.npz',coords=graph['coords'],edges=edges,shape=graph['shape'])
                    save(dest/'prediction.json',dict(scoring=stats,solver=solver,accelerator='none'))
                result.update(current_video=name,elapsed_seconds=time.monotonic()-start);save(root/'result.json',result)
                print('CPU_ASSOCIATION_VIDEO',role,name,time.monotonic()-started,flush=True)
            csvs={}
            for arm in arms:
                path=root/f'{role}_{arm}.csv';write_predictions_csv(root/role/arm,names,path);csvs[arm]=path
            save(root/f'{role}_predictions_frozen.json',dict(csv_sha256={a:sha(p) for a,p in csvs.items()},
                 role_annotations_read=False,selected_arm=selected_arm,configuration_sha256=sha(root/'frozen_configuration.json')))
            data=evaluation_dir(train,names,root/(role+'_data'));metrics={}
            for arm,path in csvs.items():
                frozen_hash=sha(path)
                metrics[arm]=evaluate_csv(path,data)
                if sha(path)!=frozen_hash:raise ValueError('CSV changed during evaluation')
                metrics[arm]['scope']='Conditional development; secondary detector trained on all 199 videos. E014 decoding calibrated on 16 videos.'
                save(root/f'{role}_{arm}_metrics.json',metrics[arm])
            if role=='calibration':
                selected_arm=max(arms,key=lambda a:metrics[a]['summary']['score'])
                selection=dict(selected_arm=selected_arm,summaries={a:m['summary'] for a,m in metrics.items()},
                    thresholds=thresholds,evaluation_annotations_read=False)
                save(root/'decoding_selection.json',selection);result['calibration']=selection
            else:
                result.update(candidate=metrics[selected_arm]['summary'],selected_arm=selected_arm,
                    candidate_csv_sha256=sha(csvs[selected_arm]),control_score=pins['paired_control_score'],
                    score_delta=metrics[selected_arm]['summary']['score']-pins['paired_control_score'])
        result.update(status='complete',seconds=time.monotonic()-start);save(root/'result.json',result)
        print('CPU_ASSOCIATION_RESULT',json.dumps(result),flush=True)
    except Exception as error:
        result.update(failed_stage=result['status'],status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
