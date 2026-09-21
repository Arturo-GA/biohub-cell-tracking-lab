import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E058',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],purpose='Exact GT-edge overlap despite equal aggregate scores, and official-matcher residual reachability. Diagnostic only; no GT-dependent fusion.')
Path('baseline/e058_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E058_AUDIT','scripts/residual_audit_runner.py',['baseline/e058_protocol.json'],['jarturo/biohub-lab-visual-reserved-cpu','jarturo/biohub-e056-evaluate-cpu','jarturo/biohub-e057-evaluate-cpu'])
