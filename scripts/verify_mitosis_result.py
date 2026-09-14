"""Audit a downloaded E007 run without rerunning image inference or changing thresholds."""
import ast
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from biohub_lab.mitosis_repair import load_specialist,repair_candidates,select_repairs
from biohub_lab.mitosis_train import E006_HASHES,ranking_metrics,operating_counts
from biohub_lab.submission import read_and_validate


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_metrics(actual,expected):
    if actual.keys()!=expected.keys(): raise ValueError('Metric fields differ')
    for key,value in actual.items():
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-12,err_msg=key)
        elif value!=expected[key]: raise ValueError(f'Metric differs: {key}')


def verify(root=Path('outputs/mitosis_train'),destination=Path('results/E007_completed.json')):
    root=Path(root);launch=read('results/E007_launch.json');status=read(root/'status.json')
    if status['status']!='COMPLETE': raise ValueError('E007 not complete')
    result=read(root/'mitosis_experiment_receipt.json');diagnostic=read(root/'mitosis_diagnostic_receipt.json')
    if result['diagnostic']!=diagnostic: raise ValueError('Inconsistent diagnostic receipts')
    notebooks=list((root/'notebook_source').glob('*.ipynb'))
    if len(notebooks)!=1: raise ValueError('Expected one downloaded notebook')
    notebook=read(notebooks[0]);code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assignment=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    if hashlib.sha256(payload).hexdigest()!=launch['payload_sha256']: raise ValueError('Notebook payload changed')
    torch.set_num_threads(2);provenance={}
    for group,training in result['groups'].items():
        folder=root/'mitosis_models'/group;split=training['split']
        if training!=read(folder/'training_receipt.json'): raise ValueError('Training receipts differ')
        if sha(folder/'best.pt')!=training['checkpoint_sha256']: raise ValueError('Specialist checksum mismatch')
        initial=Path('outputs/dense_train/temporal_models')/group/'best.pt'
        if sha(initial)!=E006_HASHES[group] or training['initial_checkpoint_sha256']!=E006_HASHES[group]:
            raise ValueError('Frozen E006 checksum mismatch')
        model,checkpoint=load_specialist(folder/'best.pt',torch.device('cpu'))
        original=torch.load(initial,map_location='cpu',weights_only=False)
        pretrained=torch.load(folder/'pretrained.pt',map_location='cpu',weights_only=False)
        if checkpoint['split']!=split or pretrained['split']!=split or original['split']!=split:
            raise ValueError('Checkpoint split mismatch')
        if checkpoint['threshold']!=training['threshold'] or checkpoint['dev']!=training['dev']:
            raise ValueError('Development operating point mismatch')
        if checkpoint['step']!=training['selected_step']: raise ValueError('Selected step mismatch')
        if pretrained['step']!=4000 or checkpoint['pretrain_steps']!=4000 or checkpoint['finetune_steps']!=3000:
            raise ValueError('Unexpected training budgets')
        history=read(folder/'history.json')
        if max(r['step'] for r in history if r['stage']=='synthetic')!=4000 or max(r['step'] for r in history if r['stage']=='real')!=3000:
            raise ValueError('Incomplete training history')
        for name,tensor in model.backbone.state_dict().items():
            torch.testing.assert_close(tensor,original['state_dict'][name],rtol=0,atol=0)
            torch.testing.assert_close(pretrained['state_dict']['backbone.'+name],tensor,rtol=0,atol=0)
        if not all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()): raise ValueError('Nonfinite weights')
        if not all(n.startswith(group+'_') for n in split['holdout']) or any(n.startswith(group+'_') for n in split['train']+split['dev']):
            raise ValueError('Embryo leakage')
        if set(split['train'])&set(split['dev']): raise ValueError('Development overlap')
        manifest=read(folder/'synthetic_manifest.json')
        if not manifest['complete'] or len(manifest['videos'])!=2048 or manifest['frames']!=9:
            raise ValueError('Incomplete synthetic data')
        if manifest['held_group']!=group or manifest['training_videos']!=split['train'] or manifest['dev_videos']!=split['dev']:
            raise ValueError('Synthetic split mismatch')
        if any(t['video'] not in split['train'] for t in manifest['template_sources']): raise ValueError('Texture leakage')
        if sum(v['positive_pairs'] for v in manifest['videos'])!=training['synthetic_training_positive']:
            raise ValueError('Synthetic positive count mismatch')
        if sum(v['negative_pairs'] for v in manifest['videos'])!=training['synthetic_training_negative']:
            raise ValueError('Synthetic negative count mismatch')
        with np.load(folder/'dev_scores.npz') as dev:
            if not set(dev['video'])<=set(split['dev']): raise ValueError('Development score leakage')
            same_metrics(ranking_metrics(dev['labels'],dev['scores']),training['dev'])
        with np.load(folder/'holdout_scores.npz') as holdout:
            if not set(holdout['video'])<=set(split['holdout']): raise ValueError('Holdout score membership mismatch')
            metrics=ranking_metrics(holdout['labels'],holdout['scores'])
            metrics={k:v for k,v in metrics.items() if not k.startswith('threshold')}
            metrics['at_dev_threshold']=operating_counts(holdout['labels'],holdout['scores'],checkpoint['threshold'])
            same_metrics(metrics,training['holdout'])
        provenance[group]=dict(checkpoint_sha256=sha(folder/'best.pt'),pretrained_sha256=sha(folder/'pretrained.pt'),
            synthetic_manifest_sha256=sha(folder/'synthetic_manifest.json'),dev_scores_sha256=sha(folder/'dev_scores.npz'),
            holdout_scores_sha256=sha(folder/'holdout_scores.npz'),weights_loaded_strictly=True,
            frozen_E006_unchanged_in_both_checkpoints=True,appearance_train_only_verified=True,
            dev_threshold_recomputed=True,holdout_metrics_recomputed_at_fixed_dev_threshold=True)
    final=root/'mitosis_diagnostic/predictions.csv'
    if sha(final)!=diagnostic['csv_sha256']: raise ValueError('CSV checksum mismatch')
    base=read_and_validate('outputs/diagnostic/biohub_control/submission.csv',diagnostic['shapes'])
    predicted=read_and_validate(final,diagnostic['shapes']);comparison={}
    for name,(nodes,edges) in predicted.items():
        bn,be=base[name];ordered=sorted(bn);lookup={node:i for i,node in enumerate(ordered)}
        if [v for _,v in sorted(nodes.items())]!=[bn[n] for n in ordered]: raise ValueError('Detector coordinates changed')
        baseline=np.array([(lookup[s],lookup[t]) for s,t in be],np.int64).reshape(-1,2)
        base_set=set(map(tuple,baseline));pred_set=set(edges)
        if not base_set<=pred_set: raise ValueError('Baseline edge removed')
        with np.load(root/'mitosis_diagnostic'/(name+'_scores.npz')) as z:
            coords=np.array([[v[k] for k in ('t','z','y','x')] for _,v in sorted(nodes.items())],np.float32)
            np.testing.assert_array_equal(coords,z['coords'])
            np.testing.assert_array_equal(baseline,z['baseline'])
            np.testing.assert_array_equal(repair_candidates(coords,baseline),z['triples'])
            threshold=result['groups'][name.split('_')[0]]['threshold']
            replay,stats,selected=select_repairs(coords,baseline,z['triples'],z['scores'],threshold)
            np.testing.assert_array_equal(replay,z['selected']);np.testing.assert_array_equal(selected,z['selected_repairs'])
            if set(map(tuple,replay))!=pred_set: raise ValueError('Saved scores do not reconstruct final CSV')
            if any(diagnostic['statistics'][name][k]!=v for k,v in stats.items()): raise ValueError('Repair statistics mismatch')
        forks=lambda es:sum(v==2 for v in Counter(a for a,b in es).values())
        comparison[name]=dict(nodes=len(nodes),baseline_edges=len(be),edges=len(edges),added_edges=len(pred_set-base_set),
            removed_edges=0,control_predicted_forks=forks(be),predicted_forks=forks(edges))
    summary=diagnostic['metrics']['summary'];control=diagnostic['control']
    if control!=read('outputs/diagnostic/run_receipt.json')['arms']['control']['metrics']['summary']:
        raise ValueError('Comparison control changed')
    same_metrics({k:summary[k]-control[k] for k in diagnostic['delta']},diagnostic['delta'])
    record=dict(experiment='E007',kernel='jarturo/biohub-lab-mitosis-specialist',version=launch['versionNumber'],
        status=status['status'],remote_status=status,payload_sha256=launch['payload_sha256'],result=result,
        verification=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),source_payload_verified=True,
            checkpoint_and_split_verification=provenance,csv_sha256=sha(final),csv_validation_passed=True,
            repaired_graph_reconstructed_from_saved_scores=True,per_video_graph_comparison=comparison,
            official_metrics_source='Recovered from the complete pinned-metric Kaggle run; image/GEFF evaluation not rerun locally.'),
        leaderboard_submitted=False,decision='do_not_submit_no_additional_division_recall_and_more_false_positives',
        reason='Preserved all baseline edges, but added 61 edges without an increase in official division TP; FP rose from 1 to 6 and score fell by 0.0108771322.')
    Path(destination).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=record['status'],score=summary['score'],delta=diagnostic['delta'],graph_comparison=comparison)))
    return record


if __name__=='__main__': verify()
