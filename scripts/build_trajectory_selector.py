"""E063: learned selection, fit/development disjoint from the reused evaluation."""
import json,sys
from pathlib import Path
from build_three_lines import build

stage=sys.argv[1];assert stage in ('DETECT','PREPARE','INFER','SELECT','EVALUATE','AUDIT')
if stage=='DETECT':
    previous=json.loads(Path('baseline/e039_protocol.json').read_text())
    cfg=dict(experiment='E063',seed=632109,fit=previous['fit'],development=previous['development'],
        evaluation=json.loads(Path('baseline/e061_protocol.json').read_text())['evaluation'],
        primary_sha256='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771',
        training_sha256=json.loads(Path('results/E061_TRAIN_completed.json').read_text())['manifest_sha256'],
        thresholds=[.05,.1,.2,.35,.5,.7,.85,.95,1.01],node_budget_fraction=.01,
        protocol='Fit trajectory quality on 40 videos, select threshold on 16 separate development videos, then freeze and evaluate on 24 reused research videos. Sparse official pred_valid edges only; unknown is never negative. Fit/dev complements use a single primary forward proxy baseline, not full Harmonic. Evaluation uses actual E062 Harmonic union. Detector fit overlaps selector fit, but not development or evaluation; public checkpoint pretraining membership unknown. CPU graph matching, model fit and calibration; GPU neural inference only. No automatic submission.')
    sets=[set(cfg[k]) for k in ('fit','development','evaluation')]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    Path('baseline/e063_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=['baseline/e063_protocol.json','scripts/residual_detector_io.py']
kernels={'DETECT':['jarturo/biohub-e061-train'], 'PREPARE':['jarturo/biohub-e063-detect'],
    'INFER':['jarturo/biohub-e063-prepare-cpu'], 'SELECT':['jarturo/biohub-e063-prepare-cpu','jarturo/biohub-e063-infer'],
    'EVALUATE':['jarturo/biohub-e063-select-cpu','jarturo/biohub-e062-prepare-cpu','jarturo/biohub-e062-infer','jarturo/biohub-e061-evaluate-cpu'],
    'AUDIT':['jarturo/biohub-e063-evaluate-cpu']}[stage]
pins={s:json.loads(Path('results/E063_'+s+'_completed.json').read_text())['manifest_sha256'] for s in {'DETECT':[], 'PREPARE':['DETECT'], 'INFER':['PREPARE'], 'SELECT':['PREPARE','INFER'],'EVALUATE':['SELECT'],'AUDIT':['EVALUATE']}[stage]}
if stage=='EVALUATE':
    pins.update({s:json.loads(Path('results/'+s+'_completed.json').read_text())['manifest_sha256'] for s in ['E062_PREPARE','E062_INFER','E061_EVALUATE']})
Path('baseline/e063_'+stage.lower()+'_pins.json').write_text(json.dumps(pins,indent=2)+'\n');files+=['baseline/e063_'+stage.lower()+'_pins.json']
if stage in ('DETECT','INFER'):files+=['src/biohub_lab/independent_expert.py']
if stage=='DETECT':files+=['src/biohub_lab/residual_detector.py','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py']
if stage in ('SELECT','EVALUATE','AUDIT'):
    files+=json.loads(Path('kaggle/e062_evaluate/payload.json').read_text())['files']
    files+=['src/biohub_lab/trajectory_selector.py','scripts/trajectory_selector_select_runner.py']
if stage=='AUDIT':files+=['scripts/residual_audit_runner.py']
build('E063_'+stage,'scripts/trajectory_selector_'+stage.lower()+'_runner.py',files,kernels,gpu=stage in ('DETECT','INFER'))
if stage in ('DETECT','INFER'):
    p=Path('kaggle/e063_'+stage.lower()+'/kernel-metadata.json');m=json.loads(p.read_text());m['competition_sources']=['biohub-cell-tracking-during-development'];m['dataset_sources']=['pilkwang/biohub-tracking-support-pack-50ep-v1'];p.write_text(json.dumps(m,indent=2)+'\n')
