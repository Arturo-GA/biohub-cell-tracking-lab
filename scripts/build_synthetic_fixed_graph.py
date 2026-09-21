import json
from pathlib import Path
from build_three_lines import build

cfg=dict(experiment='E057',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],arms=['specialist','balanced','uncertainty'],protocol='Original candidate support and nodes, original divisions locked by visual assignment. Strict equality of captured old/new detection coordinates; remap new probabilities to OLD GEFF IDs by coordinates, never assume new graph ID equality. New expert probabilities alone, log pool weight .5, or uncertainty-gated weight <=.5 when original margin<.2 and new margin exceeds original by .02. Same eight reused videos; exploratory. Frozen before observing E056 scores.')
cfg['specialist_definition']='The E056 public+synthetic fused probabilities (synthetic logit weight <=.25), not standalone synthetic-model probabilities. E057 mixes two pipelines, sharing the public models.'
Path('baseline/e057_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=json.loads(Path('kaggle/visual_validation/payload.json').read_text())['files']+['baseline/e057_protocol.json','scripts/ensemble_evaluate_runner.py','src/biohub_lab/temporal_detector.py','src/biohub_lab/association_pool.py']
build('E057_EVALUATE','scripts/synthetic_fixed_graph_runner.py',files,['jarturo/biohub-lab-visual-reserved-cpu','jarturo/biohub-lab-visual-reserved-capture','jarturo/biohub-e056-synthetic-specialist'])
