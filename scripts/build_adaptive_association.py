import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E051',arms=['mutual','adaptive','adaptive_mutual'],protocol='Frozen sparse Transformer probabilities from E023. CPU intensity/shape/radial descriptors and robust local motion. Weights from top2 margins, Transformer at least half. Mutual-best probability bonus0.25. No fitting, no embryo routing. Select best vs visual-only control before any submission.',public_sources=['yudaiyamauchi/lb-exploration-e-mutual-best','yudaiyamauchi/lb-exploration-c-disagreement-adaptive','arnav170/biohub-reid3s'])
Path('baseline/e051_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E051_EVALUATE','scripts/adaptive_association_runner.py',['baseline/e051_protocol.json','baseline/e023_validation.json','scripts/ensemble_evaluate_runner.py','src/biohub_lab/temporal_detector.py','src/biohub_lab/adaptive_association.py','src/biohub_lab/visual_assignment.py','src/biohub_lab/detection_identity.py','src/biohub_lab/association_rank.py'],['jarturo/biohub-lab-visual-reserved-cpu'])
