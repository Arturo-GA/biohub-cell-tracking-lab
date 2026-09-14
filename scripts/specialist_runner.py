"""E007: train a mitosis specialist, preserve Harmonic edges, evaluate the final CSV."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def source_root(slug,required):
    paths=[Path('/kaggle/input')/slug,Path('/kaggle/input/notebooks/jarturo')/slug]
    found=[p for p in paths if (p/required).exists()]
    if len(found)!=1: raise ValueError(f'Attach completed {slug}: {found}')
    return found[0]


def fold(cache,group,initial,output,scenes=2048,pretrain_steps=4000,finetune_steps=3000):
    from biohub_lab.temporal_train import CacheStore,make_split
    from biohub_lab.dense_lineage import build_cache
    from biohub_lab.mitosis_train import train_specialist
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    store=CacheStore(cache);split=make_split(store.names,group)
    synthetic=Path('/tmp/biohub_mitosis')/group
    manifest=build_cache(store,split,synthetic,scenes=scenes,frames=9)
    del store
    (output/'synthetic_manifest.json').write_text(json.dumps(manifest,indent=2))
    train_specialist(cache,synthetic,initial,output,group,pretrain_steps,finetune_steps)


def diagnostic():
    import numpy as np
    import torch
    from temporal_runner import competition
    from biohub_lab.submission import COLUMNS,read_and_validate
    from biohub_lab.evaluate import evaluate_csv,shapes_for
    from biohub_lab.mitosis_repair import load_specialist,repair_video
    previous=source_root('biohub-lab-official-metric-ab','run_receipt.json')
    receipt=json.loads((previous/'run_receipt.json').read_text())
    control_csv=previous/'biohub_control/submission.csv'
    if hashlib.sha256(control_csv.read_bytes()).hexdigest()!=receipt['arms']['control']['sha256']:
        raise ValueError('Harmonic control checksum mismatch')
    root=Path('/kaggle/working/mitosis_diagnostic');root.mkdir(exist_ok=True)
    data=root/'data';data.mkdir(exist_ok=True)
    for name in receipt['datasets']:
        for suffix in ('.zarr','.geff'):
            destination=data/(name+suffix)
            if not destination.exists(): destination.symlink_to(competition()/'train'/(name+suffix),target_is_directory=True)
    shapes=shapes_for(data);base=read_and_validate(control_csv,shapes)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    models={};stats={};index=0;final=root/'predictions.csv'
    with final.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(COLUMNS)
        for name,(nodes,edges) in sorted(base.items()):
            group=name.split('_')[0]
            if group not in models:
                path=Path('/kaggle/working/mitosis_models')/group
                training=json.loads((path/'training_receipt.json').read_text())
                if hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()!=training['checkpoint_sha256']:
                    raise ValueError('Specialist checkpoint checksum mismatch')
                models[group]=load_specialist(path/'best.pt',device)
            model,checkpoint=models[group]
            if name not in checkpoint['split']['holdout'] or any(n.startswith(group+'_') for n in checkpoint['split']['train']+checkpoint['split']['dev']):
                raise ValueError('Specialist trained or calibrated with diagnostic embryo')
            ordered=sorted(nodes);lookup={node:i for i,node in enumerate(ordered)}
            coords=np.array([[nodes[n][k] for k in ('t','z','y','x')] for n in ordered],np.float32)
            old=np.array([(lookup[s],lookup[t]) for s,t in edges],np.int64).reshape(-1,2)
            start=time.monotonic()
            selected,stat,arrays=repair_video(model,checkpoint,coords,old,data/(name+'.zarr'),device)
            if not set(map(tuple,old))<=set(map(tuple,selected)): raise ValueError('Control edges not preserved')
            stat.update(seconds=time.monotonic()-start,nodes=len(coords),edges=len(selected))
            stats[name]=stat
            np.savez_compressed(root/(name+'_scores.npz'),coords=coords,baseline=old,selected=selected,**arrays)
            for node,(t,z,y,x) in enumerate(coords):
                writer.writerow([index,name,'node',node,int(t),int(z),int(y),int(x),-1,-1]);index+=1
            for s,t in selected:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(s),int(t)]);index+=1
            print('MITOSIS_REPAIR',name,json.dumps(stat),flush=True)
    parsed=read_and_validate(final,shapes)
    for name,(nodes,_) in parsed.items():
        if [v for _,v in sorted(nodes.items())]!=[v for _,v in sorted(base[name][0].items())]:
            raise ValueError('Detector coordinates changed')
    metrics=evaluate_csv(final,data)
    metrics['scope']='Specialist and threshold exclude each diagnostic embryo; the unchanged public detector/linker was trained on these videos. Conditional comparison, not independent end-to-end validation.'
    control=receipt['arms']['control']['metrics']['summary']
    result=dict(metrics=metrics,control=control,statistics=stats,shapes=shapes,
        delta={k:metrics['summary'][k]-control[k] for k in ('score','adj_edge_jaccard','division_jaccard')},
        csv_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),validated=True,
        all_baseline_edges_preserved=True,all_detector_coordinates_preserved=True,
        leaderboard_submitted=False,threshold_selection='Known training-embryo development triples only; no diagnostic threshold sweep')
    (root/'official_metrics.json').write_text(json.dumps(metrics,indent=2))
    Path('/kaggle/working/mitosis_diagnostic_receipt.json').write_text(json.dumps(result,indent=2))
    print('MITOSIS_DIAGNOSTIC_COMPLETE',json.dumps(result['delta']),flush=True)
    return result


def main(package):
    import torch
    from temporal_runner import input_cache
    from biohub_lab.mitosis_train import E006_HASHES
    start=time.monotonic();cache=input_cache()
    previous=source_root('biohub-lab-dense-lineage-pretraining','dense_experiment_receipt.json')
    root=Path('/kaggle/working/mitosis_models');processes=[]
    for i,group in enumerate(('44b6','6bba')):
        initial=previous/'temporal_models'/group/'best.pt'
        if hashlib.sha256(initial.read_bytes()).hexdigest()!=E006_HASHES[group]:
            raise ValueError('Unexpected E006 model version')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%max(torch.cuda.device_count(),1)))
        command=[sys.executable,'-u',str(Path(package)/'scripts/specialist_runner.py'),'fold',str(cache),group,str(initial),str(root/group)]
        if torch.cuda.device_count()>1: processes.append(subprocess.Popen(command,env=env))
        else: subprocess.run(command,env=env,check=True)
    for process in processes:
        if process.wait()!=0: raise RuntimeError('Mitosis specialist fold failed')
    groups={g:json.loads((root/g/'training_receipt.json').read_text()) for g in ('44b6','6bba')}
    result=diagnostic()
    receipt=dict(experiment='E007',groups=groups,diagnostic=result,seconds=time.monotonic()-start,
        design='Train a new six-timepoint event network with frozen E006 features. Preserve all Harmonic nodes/edges; assign at most one orphan daughter to a one-child mother.',
        leaderboard_submitted=False,decision='review_complete_results_before_any_test_run')
    Path('/kaggle/working/mitosis_experiment_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('MITOSIS_EXPERIMENT_COMPLETE',json.dumps(result['delta']),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='fold': fold(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    elif sys.argv[1]=='specialist': main(sys.argv[2])
    else: raise ValueError(sys.argv[1])
