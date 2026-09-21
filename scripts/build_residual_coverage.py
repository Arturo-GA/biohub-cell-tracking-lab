import json
from pathlib import Path
from build_three_lines import build
files=json.loads(Path('kaggle/e061_evaluate/payload.json').read_text())['files']
build('E061_COVERAGE','scripts/residual_detector_coverage_runner.py',files,[
    'jarturo/biohub-e061-infer','jarturo/biohub-temporal-graph-prepare-cpu','jarturo/biohub-lab-visual-reserved-cpu'])
