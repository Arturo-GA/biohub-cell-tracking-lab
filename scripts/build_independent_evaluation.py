import json
from pathlib import Path
from build_three_lines import build

files=json.loads(Path('kaggle/visual_validation/payload.json').read_text())['files']+[
    'baseline/e059_protocol.json','scripts/ensemble_evaluate_runner.py',
    'src/biohub_lab/temporal_detector.py','src/biohub_lab/association_pool.py',
    'scripts/residual_audit_runner.py']
build('E059_EVALUATE','scripts/independent_evaluate_runner.py',files,[
    'jarturo/biohub-lab-visual-reserved-cpu',
    'jarturo/biohub-lab-visual-reserved-capture','jarturo/biohub-e059-capture'])
