"""E006: dense pretraining, unchanged real fine-tuning, and automatic evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def fold(cache,group,output,scenes=2048,pretrain_steps=4000,real_steps=3000):
    from biohub_lab.temporal_train import CacheStore,make_split,train_fold
    from biohub_lab.dense_lineage import build_cache
    from biohub_lab.dense_pretrain import pretrain
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    store=CacheStore(cache);split=make_split(store.names,group)
    synthetic=Path('/tmp/biohub_dense')/group
    generated=build_cache(store,split,synthetic,scenes=scenes)
    (root/'synthetic_manifest.json').write_text(json.dumps(generated,indent=2))
    for p in synthetic.glob('preview_*.npz'): shutil.copyfile(p,root/p.name)
    del store
    pretrained=pretrain(synthetic,root,steps=pretrain_steps)
    train_fold(cache,root,group,steps=real_steps,initial_checkpoint=pretrained)
    print('DENSE_FOLD_COMPLETE',group,flush=True)


def main(package):
    import torch
    from temporal_runner import input_cache,diagnostic
    cache=input_cache();start=time.monotonic();root=Path('/kaggle/working/temporal_models')
    processes=[]
    for i,group in enumerate(('44b6','6bba')):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%max(torch.cuda.device_count(),1)))
        command=[sys.executable,'-u',str(Path(package)/'scripts/dense_runner.py'),'fold',str(cache),group,str(root/group)]
        if torch.cuda.device_count()>1: processes.append(subprocess.Popen(command,env=env))
        else: subprocess.run(command,env=env,check=True)
    for p in processes:
        if p.wait()!=0: raise RuntimeError('Dense pretraining/fine-tuning worker failed')
    groups={}
    for group in ('44b6','6bba'):
        path=root/group
        training=json.loads((path/'training_receipt.json').read_text())
        groups[group]=dict(training=training,pretraining=json.loads((path/'pretrain_receipt.json').read_text()),
            synthetic_manifest_sha256=hashlib.sha256((path/'synthetic_manifest.json').read_bytes()).hexdigest())
    diagnostic(package)
    current=json.loads(Path('/kaggle/working/temporal_diagnostic_receipt.json').read_text())
    previous=next(p for p in [Path('/kaggle/input/biohub-lab-temporal-train'),
        Path('/kaggle/input/notebooks/jarturo/biohub-lab-temporal-train')] if (p/'temporal_training_receipt.json').exists())
    old=json.loads((previous/'temporal_diagnostic_receipt.json').read_text())
    old_training=json.loads((previous/'temporal_training_receipt.json').read_text())
    from biohub_lab.temporal_submission import CHECKPOINT_HASHES
    for group,expected in CHECKPOINT_HASHES.items():
        if old_training[group]['checkpoint_sha256']!=expected:
            raise ValueError('E004 reference changed; refuse an unpinned comparison')
    receipt=dict(experiment='E006',groups=groups,seconds=time.monotonic()-start,
        comparison_to_E004={k:current['metrics']['summary'][k]-old['metrics']['summary'][k]
            for k in ('score','adj_edge_jaccard','division_jaccard')},
        metrics=current['metrics'],comparison_to_harmonic=current['delta'],
        heldout_comparison={g:dict(
            parent_accuracy_delta=groups[g]['training']['holdout']['summary']['parent_accuracy']-old_training[g]['holdout']['summary']['parent_accuracy'],
            division_ap_delta=groups[g]['training']['holdout']['divisions']['average_precision']-old_training[g]['holdout']['divisions']['average_precision'])
            for g in groups},
        design='Same temporal architecture, real fine-tuning batches, checkpoint criterion, inference priors and thresholds as E004; only initialization changed through dense synthetic pretraining.',
        scope='Embryo-held-out temporal models; the four-video end-to-end diagnostic is conditional on a public detector trained on those videos.',
        leaderboard_submitted=False,decision='review_complete_results_before_test_inference')
    Path('/kaggle/working/dense_experiment_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('DENSE_EXPERIMENT_COMPLETE',json.dumps(receipt['comparison_to_E004']),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='fold': fold(sys.argv[2],sys.argv[3],sys.argv[4])
    elif sys.argv[1]=='dense': main(sys.argv[2])
    else: raise ValueError(sys.argv[1])
