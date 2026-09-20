"""Freeze the two subsequent experiments. Launches remain strictly sequential."""
import json
from pathlib import Path
from build_three_lines import build
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'baseline/e038_protocol.json').read_text());prep=json.loads((ROOT/'results/E038_PREPARE_completed.json').read_text())
    c39=dict(experiment='E039',seed=390920,steps=2000,batch=4,deadline_seconds=2400,fit=cfg['fit'],development=cfg['development'],data_manifest_sha256=prep['manifest_sha256'],frames=[10,30,50,70,90],budgets=[128,256,512],protocol='Paired 1-frame and 3-frame fully convolutional 3D center-vector fields trained from scratch on identical sparse annotations and sampling seeds. Supervision restricted to within3 isotropic voxels of known centers, all other voxels masked. Bounded displacement4 voxels, SmoothL1, AdamW3e-4, 2000 updates/arm. Dense heldout inference on fixed5 frames/video. Image-weighted trilinear voting; compare equal candidate budgets with image DoG peaks. Primary gate at256: improve3um recall vs static and DoG in both embryos with no7um recall loss. No threshold sweep or submission; sparse recall cannot establish precision.')
    p=ROOT/'baseline/e039_protocol.json';assert not p.exists();p.write_text(json.dumps(c39,indent=2)+'\n')
    common=['baseline/e039_protocol.json','src/biohub_lab/temporal_detector.py','src/biohub_lab/spatial_probe.py','scripts/spatial_probe_features_runner.py']
    build('E039_TRAIN','scripts/temporal_detector_train_runner.py',common.copy(),['jarturo/biohub-e038-prepare-cpu'],gpu=True)
    build('E039_INFER','scripts/temporal_detector_infer_runner.py',common.copy(),['jarturo/biohub-e038-prepare-cpu','jarturo/biohub-e039-train'],gpu=True)
    build('E039_EVALUATE','scripts/temporal_detector_evaluate_runner.py',common.copy(),['jarturo/biohub-e038-prepare-cpu','jarturo/biohub-e039-infer'])
    c40=dict(experiment='E040',seed=400920,steps=2000,batch=16,deadline_seconds=1800,fit=cfg['expanded_fit'],development=cfg['development'],protocol='Expanded real divisions:159 fit videos with124 annotated events,16 development videos with15. Preserve original calibration and old8 development outside training. Include all true binary consecutive events, no full-track-history requirement. Negative pair requires at least one daughter with a known different parent. Six nearest known daughters within20um plus own children; sample max128 negative triples/fit video. Joint five-frame32cube image and symmetric mother/daughter Gaussian query masks; train7-channel3D CNN and geometry11D head. Balanced positive/negative training, no development labels during GPU. CPU same-data ExtraTrees geometry control, repeated-center test-time ablation. Gate3TP0FP and AP+0.05 vs both controls; no automatic leaderboard submission.')
    p=ROOT/'baseline/e040_protocol.json';assert not p.exists();p.write_text(json.dumps(c40,indent=2)+'\n')
    common=['baseline/e040_protocol.json','src/biohub_lab/expanded_events.py','src/biohub_lab/temporal_data.py','src/biohub_lab/spatial_probe.py','scripts/spatial_probe_features_runner.py']
    build('E040_PREPARE','scripts/expanded_events_prepare_runner.py',common.copy(),['jarturo/biohub-e038-prepare-cpu'])
    build('E040_TRAIN','scripts/expanded_events_train_runner.py',common.copy(),['jarturo/biohub-e040-prepare-cpu'],gpu=True)
    build('E040_EVALUATE','scripts/expanded_events_evaluate_runner.py',common.copy(),['jarturo/biohub-e040-prepare-cpu','jarturo/biohub-e040-train'])
if __name__=='__main__':main()
