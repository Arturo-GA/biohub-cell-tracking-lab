import json,sys
from pathlib import Path
from build_three_lines import build
stage=sys.argv[1];assert stage in ('PREPARE','INFER','EVALUATE')
if stage=='PREPARE':
    cfg=dict(experiment='E062',videos=json.loads(Path('baseline/e061_protocol.json').read_text())['evaluation'],
        predictions_sha256=json.loads(Path('results/E061_INFER_completed.json').read_text())['manifest_sha256'],
        evaluation_sha256=json.loads(Path('results/E061_EVALUATE_completed.json').read_text())['manifest_sha256'],
        primary_sha256='12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771',
        protocol='Replace hard nearest/intensity persistence with pretrained primary UNet/Transformer association on union of base nodes and adapted detector candidates >3um from base. Original nodes and edges remain fixed in final ensemble. Primary forward neural probabilities, not a reproduction of full harmonic dual-seed/TTA fusion. Top8 row/column edges, global per-frame assignment log(p/.05), maximum14um, chains >=5. Same E061 addition/anchor/confidence limits; no threshold sweep. 24 reused research movies. GPU only new neural associations; CPU union and evaluation.')
    Path('baseline/e062_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=['baseline/e062_protocol.json','scripts/residual_detector_io.py','src/biohub_lab/submission.py']
kernels={'PREPARE':['jarturo/biohub-e061-infer','jarturo/biohub-e061-evaluate-cpu'],
    'INFER':['jarturo/biohub-e062-prepare-cpu'],
    'EVALUATE':['jarturo/biohub-e062-prepare-cpu','jarturo/biohub-e062-infer','jarturo/biohub-e061-evaluate-cpu']}[stage]
if stage=='INFER':
    r=json.loads(Path('results/E062_PREPARE_completed.json').read_text());Path('baseline/e062_infer_pins.json').write_text(json.dumps(dict(prepared_sha256=r['manifest_sha256']),indent=2)+'\n')
    files+=['baseline/e062_infer_pins.json','src/biohub_lab/independent_expert.py']
if stage=='EVALUATE':
    r=json.loads(Path('results/E062_INFER_completed.json').read_text());Path('baseline/e062_evaluate_pins.json').write_text(json.dumps(dict(inference_sha256=r['manifest_sha256']),indent=2)+'\n')
    files+=json.loads(Path('kaggle/e061_evaluate/payload.json').read_text())['files']
    files+=['baseline/e062_evaluate_pins.json','src/biohub_lab/neural_complement.py']
build('E062_'+stage,'scripts/neural_complement_'+stage.lower()+'_runner.py',files,kernels,gpu=stage=='INFER')
if stage=='INFER':
    p=Path('kaggle/e062_infer/kernel-metadata.json');m=json.loads(p.read_text());m['competition_sources']=['biohub-cell-tracking-during-development'];m['dataset_sources']=['pilkwang/biohub-tracking-support-pack-50ep-v1'];p.write_text(json.dumps(m,indent=2)+'\n')
