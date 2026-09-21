import json
from pathlib import Path
from build_three_lines import build

cfg=json.loads(Path('baseline/e059_protocol.json').read_text())
cfg['experiment']='E060'
cfg['protocol']='Apply atomic disagreement components from standalone SWA/fold0 to visual only if expert likelihood gain per changed edge >= log(2). Joint additionally requires complete temporal context, curvature <=4um and <= old+.5um. Thresholds inherited unchanged from E055, not fitted to E059 rescued labels. Four variants, fixed original nodes/divisions. Reused eight videos, exploratory. Labels read only after CSV hashes frozen.'
Path('baseline/e060_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=json.loads(Path('kaggle/e059_evaluate/payload.json').read_text())['files']+[
    'baseline/e060_protocol.json','src/biohub_lab/selective_repair.py']
build('E060_EVALUATE','scripts/selective_independent_runner.py',files,[
    'jarturo/biohub-lab-visual-reserved-cpu',
    'jarturo/biohub-lab-visual-reserved-capture','jarturo/biohub-e059-capture'])
