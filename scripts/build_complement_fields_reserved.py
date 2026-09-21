import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E053',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],dense_manifest_sha256=json.loads(Path('results/E041_TRAIN_completed.json').read_text())['manifest_sha256'],static_manifest_sha256=json.loads(Path('results/E039_TRAIN_completed.json').read_text())['manifest_sha256'],protocol='Frozen neural fields only on GPU. CPU evaluates dense/weighted bridges with original and visual associations under the public postprocess configurations. Eight reused videos, exploratory.')
Path('baseline/e053_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E053_FIELDS','scripts/complement_fields_reserved_runner.py',['baseline/e053_protocol.json','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py','scripts/ensemble_evaluate_runner.py'],['jarturo/biohub-e041-train','jarturo/biohub-e039-train'],gpu=True)
p=Path('kaggle/e053_fields/kernel-metadata.json');m=json.loads(p.read_text());m['competition_sources']=['biohub-cell-tracking-during-development'];p.write_text(json.dumps(m,indent=2)+'\n')
