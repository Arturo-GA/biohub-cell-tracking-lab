"""Prepare and train the temporal model; every stage records scope and provenance."""
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess

from biohub_lab.temporal_data import prepare_video


def competition():
    return next(p for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
        Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())


def prepare():
    comp=competition(); root=Path('/kaggle/working/temporal_cache')
    paths=sorted((comp/'train').glob('*.zarr'))
    start=time.monotonic(); receipts=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(prepare_video,p,root/p.stem):p for p in paths}
        for future in as_completed(pending):
            r=future.result(); receipts.append(r)
            print('CACHE',len(receipts),'/',len(paths),json.dumps(r),flush=True)
            (root/'cache_manifest.json').write_text(json.dumps(dict(videos=receipts,complete=False),indent=2))
    (root/'cache_manifest.json').write_text(json.dumps(dict(videos=sorted(receipts,key=lambda r:r['name']),
        complete=True,seconds=time.monotonic()-start,scope='Raw image patches and sparse GT; no pretrained model outputs'),indent=2))
    print('CACHE_COMPLETE',len(receipts),'seconds',time.monotonic()-start,flush=True)


def input_cache():
    roots=list(Path('/kaggle/input').glob('biohub-lab-temporal-prepare/temporal_cache'))
    roots+=list(Path('/kaggle/input/notebooks/jarturo/biohub-lab-temporal-prepare/temporal_cache').parent.glob('temporal_cache'))
    matches=[p for p in roots if (p/'cache_manifest.json').exists()]
    if len(matches)!=1: raise ValueError(f'Attach complete Temporal Prepare output: {matches}')
    return matches[0]


def train(package):
    import torch
    cache=input_cache(); manifest=json.loads((cache/'cache_manifest.json').read_text())
    groups=sorted({r['group'] for r in manifest['videos']})
    if len(groups)!=2: raise ValueError(f'Review acquisition split for groups {groups}')
    processes=[]
    for i,group in enumerate(groups):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%max(torch.cuda.device_count(),1)))
        command=[sys.executable,'-u',str(Path(package)/'scripts/temporal_runner.py'),'fold',str(cache),group]
        if torch.cuda.device_count()>1:
            processes.append(subprocess.Popen(command,env=env))
        else: subprocess.run(command,env=env,check=True)
    for process in processes:
        if process.wait()!=0: raise RuntimeError('Training fold failed')
    receipt={g:json.loads((Path('/kaggle/working/temporal_models')/g/'training_receipt.json').read_text()) for g in groups}
    Path('/kaggle/working/temporal_training_receipt.json').write_text(json.dumps(receipt,indent=2))
    diagnostic(package)


def diagnostic(package):
    import hashlib
    import numpy as np
    import torch
    import csv
    from biohub_lab.evaluate import shapes_for,evaluate_csv
    from biohub_lab.submission import read_and_validate,COLUMNS
    from biohub_lab.temporal_inference import load_model,score_video
    candidates=[Path('/kaggle/input/biohub-lab-official-metric-ab'),
        Path('/kaggle/input/notebooks/jarturo/biohub-lab-official-metric-ab')]
    previous=next(p for p in candidates if (p/'run_receipt.json').exists())
    original=json.loads((previous/'run_receipt.json').read_text())
    seed_csv=previous/'biohub_control/submission.csv'
    if hashlib.sha256(seed_csv.read_bytes()).hexdigest()!=original['arms']['control']['sha256']:
        raise ValueError('Cached detector output checksum mismatch')
    root=Path('/kaggle/working/temporal_diagnostic'); root.mkdir(exist_ok=True)
    data=root/'data'; data.mkdir(exist_ok=True)
    for name in original['datasets']:
        for suffix in ('.zarr','.geff'):
            dest=data/(name+suffix)
            if not dest.exists(): dest.symlink_to(competition()/'train'/(name+suffix),target_is_directory=True)
    shapes=shapes_for(data); seeds=read_and_validate(seed_csv,shapes)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(2 if device.type=='cuda' else 4)
    models={}; stats={}; predictions={}
    for name,(nodes,_) in sorted(seeds.items()):
        group=name.split('_')[0]
        if group not in models:
            models[group]=load_model(Path('/kaggle/working/temporal_models')/group/'best.pt',device)
        model,checkpoint=models[group]
        if name not in checkpoint['split']['holdout'] or any(n.startswith(group+'_') for n in checkpoint['split']['train']):
            raise ValueError('Diagnostic embryo was used to train the temporal model')
        coords=np.array([[n[k] for k in ('t','z','y','x')] for _,n in sorted(nodes.items())],np.float32)
        start=time.monotonic()
        edges,stat,arrays=score_video(model,checkpoint,coords,data/(name+'.zarr'),device)
        stat['seconds']=time.monotonic()-start; stats[name]=stat; predictions[name]=(coords,edges)
        np.savez_compressed(root/(name+'_scores.npz'),coords=coords,selected=edges,**arrays)
        print('TEMPORAL_DENSE',name,json.dumps({k:v for k,v in stat.items() if k!='solver'}),flush=True)
    destination=root/'predictions.csv'
    with destination.open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(COLUMNS); index=0
        for name,(coords,edges) in sorted(predictions.items()):
            for n,(t,z,y,x) in enumerate(coords):
                writer.writerow([index,name,'node',n,int(t),int(z),int(y),int(x),-1,-1]); index+=1
            for s,t in edges:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(s),int(t)]); index+=1
    metrics=evaluate_csv(destination,data)
    metrics['scope']='Temporal encoder/linker held out by embryo; fixed public detector trained on these videos. Conditional comparison, not independent end-to-end validation.'
    control=original['arms']['control']['metrics']['summary']
    receipt=dict(metrics=metrics,control=control,statistics=stats,leaderboard_submitted=False,
        delta={k:metrics['summary'][k]-control[k] for k in ('score','adj_edge_jaccard','division_jaccard')},
        submission_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        decision='review_after_complete_oof_and_conditional_results')
    (root/'official_metrics.json').write_text(json.dumps(metrics,indent=2))
    Path('/kaggle/working/temporal_diagnostic_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('TEMPORAL_DIAGNOSTIC_COMPLETE',json.dumps(receipt['delta']),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1] in ('train','train_cpu'): train(sys.argv[2])
    elif sys.argv[1]=='diagnostic': diagnostic(sys.argv[2])
    elif sys.argv[1]=='fold':
        from biohub_lab.temporal_train import train_fold
        train_fold(sys.argv[2],Path('/kaggle/working/temporal_models')/sys.argv[3],sys.argv[3])
    else: raise ValueError(sys.argv[1])
