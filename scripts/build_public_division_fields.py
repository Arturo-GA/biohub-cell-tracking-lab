import json
from pathlib import Path
from build_three_lines import build
cfg=dict(experiment='E052',videos=json.loads(Path('baseline/e023_validation.json').read_text())['videos'],checkpoint_sha256='8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0',protocol='Same public DeepCenter float32 eight-view TTA on all frames of reused E023 eight videos. Cache heatmaps only. CPU replays full postprocessor at threshold .25/.20, plus selected E051 association. No annotation read during inference.')
Path('baseline/e052_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
build('E052_INFER','scripts/public_division_fields_runner.py',['baseline/e052_protocol.json','baseline/harmonic_inference.py','src/biohub_lab/public_postprocess.py','scripts/ensemble_evaluate_runner.py','src/biohub_lab/temporal_detector.py'],[],gpu=True)
p=Path('kaggle/e052_infer/kernel-metadata.json');meta=json.loads(p.read_text());meta.update(competition_sources=['biohub-cell-tracking-during-development'],dataset_sources=['pilkwang/biohub-deepcenter-unet3d-center-prior-v1']);p.write_text(json.dumps(meta,indent=2)+'\n')
