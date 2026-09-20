import json
from pathlib import Path
from build_three_lines import build
r=Path('.')
cfg=dict(experiment='E044',budget=256,frames=[10,30,50,70,90],fusion='Equal-rank interleave dense first; reject candidate within <=1 isotropic voxel of earlier point; stop at 256; inputs top512 each. No GT candidate selection.',gate='Per embryo: ensemble matched3 >= best component and matched7 > best component. Sparse component gate only, not submission approval.',sources={})
for name,stage,path in [('dense','E041_TRAIN','dense_supervision_training/result.json'),('static','E039_INFER','temporal_detector_fields/result.json'),('raw','E038_PREPARE','three_lines_data/result.json')]:
    cfg['sources'][name]=dict(path=path,sha256=json.loads((r/f'results/{stage}_completed.json').read_text())['manifest_sha256'])
(r/'baseline/e044_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E044_EVALUATE','scripts/ensemble_evaluate_runner.py',['baseline/e044_protocol.json','src/biohub_lab/temporal_detector.py'],['jarturo/biohub-e041-train','jarturo/biohub-e039-infer','jarturo/biohub-e038-prepare-cpu'])
