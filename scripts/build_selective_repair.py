import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E055',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],arms=['confidence','trajectory','joint'],protocol='Atomic symmetric-difference components between frozen Harmonic and visual graph. Confidence requires average log gain >=log(2). Trajectory gate uses original unambiguous previous/next links: mean curvature <=4um and at most .5um worse than original. No labels or video names in routing. Existing divisions untouched.')
Path('baseline/e055_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E055_EVALUATE','scripts/selective_repair_runner.py',['baseline/e055_protocol.json','scripts/ensemble_evaluate_runner.py','src/biohub_lab/temporal_detector.py','src/biohub_lab/selective_repair.py','src/biohub_lab/visual_assignment.py','src/biohub_lab/detection_identity.py','src/biohub_lab/association_rank.py'],['jarturo/biohub-lab-visual-reserved-cpu'])
