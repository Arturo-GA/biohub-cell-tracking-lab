import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E049',graphs_sha256='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22',donors_sha256=json.loads(Path('results/E047_RECOVER_completed.json').read_text())['manifest_sha256'],selection='Best full graph local score among four prespecified variants; tie fewer FP. Exploratory submission authorized even for small gain. No embryo-specific routing.',public_idea='Robust local median motion from haideptry/biohub-sota-0-949-local-motion-flow-2xt4-19m; own reliability weighting implementation.')
Path('baseline/e049_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E049_EVALUATE','scripts/weighted_bridge_runner.py',['baseline/e049_protocol.json','scripts/ensemble_evaluate_runner.py','src/biohub_lab/temporal_detector.py','src/biohub_lab/temporal_bridge.py','src/biohub_lab/weighted_bridge.py'],['jarturo/biohub-e047-recover-cpu','jarturo/biohub-temporal-graph-prepare-cpu'])
