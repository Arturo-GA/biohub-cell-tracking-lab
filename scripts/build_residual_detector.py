import json,sys
from pathlib import Path
from build_three_lines import build

stage=sys.argv[1]
assert stage in ('PREPARE','TRAIN','INFER','EVALUATE')
if stage=='PREPARE':
    cfg=dict(experiment='E061',seed=612109,fit=json.loads(Path('baseline/e039_protocol.json').read_text())['fit'],
        evaluation=json.loads(Path('baseline/e031_dense_detector.json').read_text())['videos']+json.loads(Path('baseline/e023_validation.json').read_text())['videos'],
        crops_per_video=128,steps=2500,batch_real=4,batch_external=4,learning_rate=.0001,deadline_seconds=2400,
        real_data_sha256=json.loads(Path('baseline/e041_protocol.json').read_text())['biohub_data_sha256'],
        external_data_sha256=json.loads(Path('baseline/e041_protocol.json').read_text())['data_sha256'],
        pretrained_manifest_sha256=json.loads(Path('results/E041_TRAIN_completed.json').read_text())['manifest_sha256'],
        protocol='Adapt NIS3D DenseCenter to three real frames. Initialize temporal stem with central pretrained weights and zero side weights. Known-positive heatmap neighborhoods only, hard-positive weighting from frozen teacher, soft teacher preservation outside labeled neighborhoods, replay external fully supervised crops. Fixed 2500 updates; no evaluation data in training. Evaluate all frames of 24 reused videos disjoint from fit40. Frozen and adapted candidates, then CPU temporal confirmation and full official graph metrics. No claim of independent project holdout. No automatic submission.',
        proposals=dict(threshold=.2,topk=512),
        promotion=dict(minimum_mean_gain=.001,each_cohort_gain_nonnegative=True,edge_tp_gain_positive=True,division_fp_must_not_increase=True))
    assert not set(cfg['fit'])&set(cfg['evaluation']) and len(set(cfg['evaluation']))==24
    Path('baseline/e061_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=['baseline/e061_protocol.json','src/biohub_lab/residual_detector.py','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py','scripts/ensemble_evaluate_runner.py']
kernels={'PREPARE':['jarturo/biohub-e038-prepare-cpu'],
    'TRAIN':['jarturo/biohub-e061-prepare-cpu','jarturo/biohub-e041-train','jarturo/biohub-e041-regrid-cpu'],
    'INFER':['jarturo/biohub-e061-train','jarturo/biohub-e041-train'],
    'EVALUATE':['jarturo/biohub-e061-infer','jarturo/biohub-temporal-graph-prepare-cpu','jarturo/biohub-lab-visual-reserved-cpu']}[stage]
gpu=stage in ('TRAIN','INFER')
if stage in ('TRAIN','INFER'):
    files.remove('scripts/ensemble_evaluate_runner.py');files+=['scripts/residual_detector_io.py']
if stage=='TRAIN':
    r=json.loads(Path('results/E061_PREPARE_completed.json').read_text())
    Path('baseline/e061_train_pins.json').write_text(json.dumps(dict(prepared_sha256=r['manifest_sha256']),indent=2)+'\n')
    files+=['baseline/e061_train_pins.json']
if stage=='INFER':
    r=json.loads(Path('results/E061_TRAIN_completed.json').read_text())
    Path('baseline/e061_infer_pins.json').write_text(json.dumps(dict(training_sha256=r['manifest_sha256']),indent=2)+'\n')
    files+=['baseline/e061_infer_pins.json']
if stage=='EVALUATE':
    files+=json.loads(Path('kaggle/visual_validation/payload.json').read_text())['files']
    files+=['src/biohub_lab/residual_tracklets.py','scripts/residual_detector_io.py']
    r=json.loads(Path('results/E061_INFER_completed.json').read_text())
    Path('baseline/e061_evaluate_pins.json').write_text(json.dumps(dict(predictions_sha256=r['manifest_sha256'],graphs_sha256='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22'),indent=2)+'\n')
    files+=['baseline/e061_evaluate_pins.json']
build('E061_'+stage,'scripts/residual_detector_'+stage.lower()+'_runner.py',files,kernels,gpu=gpu)
if stage=='INFER':
    p=Path('kaggle/e061_infer/kernel-metadata.json');m=json.loads(p.read_text());m['competition_sources']=['biohub-cell-tracking-during-development'];p.write_text(json.dumps(m,indent=2)+'\n')
