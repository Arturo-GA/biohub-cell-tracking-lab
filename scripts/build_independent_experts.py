import json
from pathlib import Path
from build_three_lines import build

cfg=dict(experiment='E059',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],models=dict(swa=dict(file='synthetic_5fold_swa.pth',sha256='0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9'),fold0=dict(file='synthetic_fold0_best.pth',sha256='5d50e387d70895430a9f48093cc25dc8db7f873571e905ae5ea59666d5773bca')),protocol='Standalone neural probabilities for all frozen original detections, FP32, no TTA, no old-confidence gate. Preserve original candidate probabilities plus new top8 row/column support. Official residual audit informs interpretation but does not select videos or inference locations. Eight reused videos, exploratory; public real pretraining membership unknown.')
Path('baseline/e059_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E059_CAPTURE','scripts/independent_experts_runner.py',['baseline/e059_protocol.json','src/biohub_lab/independent_expert.py'],['jarturo/biohub-lab-visual-reserved-capture'],gpu=True)
p=Path('kaggle/e059_capture/kernel-metadata.json');m=json.loads(p.read_text());m['competition_sources']=['biohub-cell-tracking-during-development'];m['dataset_sources']=['pilkwang/biohub-tracking-support-pack-50ep-v1','bhpepper/biohub-synthetic-5fold-ensemble-v1'];p.write_text(json.dumps(m,indent=2)+'\n')
