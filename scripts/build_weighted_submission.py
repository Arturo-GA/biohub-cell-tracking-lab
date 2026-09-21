import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    r=json.loads((ROOT/'results/E049_EVALUATE_completed.json').read_text())
    mode=r['result']['selected']
    cfg=dict(experiment='E050',mode=mode,selection_receipt_sha256=sha(ROOT/'results/E049_EVALUATE_completed.json'),
        dense_manifest_sha256=json.loads((ROOT/'results/E041_TRAIN_completed.json').read_text())['manifest_sha256'],
        static_manifest_sha256=json.loads((ROOT/'results/E039_TRAIN_completed.json').read_text())['manifest_sha256'],
        public_config={'MOTION_RELINK_TIGHT_UM':5.5},public_source='raunakdey07/biohub-harmonic-fusion-v3',
        inference='All test images recomputed; fixed public tight55 plus selected E049 complement. No training data read. Exploratory transfer to public configuration, not separately validated locally.')
    (ROOT/'baseline/e050_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    names=['NOTICE.md','baseline/e050_submission.json','baseline/harmonic_inference.py','src/biohub_lab/__init__.py','src/biohub_lab/patch.py','src/biohub_lab/dense_center.py','src/biohub_lab/temporal_detector.py','src/biohub_lab/detector_complements.py','src/biohub_lab/temporal_bridge.py','src/biohub_lab/weighted_bridge.py','src/biohub_lab/submission.py','src/biohub_lab/calibration_export.py','scripts/ensemble_evaluate_runner.py','scripts/weighted_submission_runner.py','scripts/weighted_submission_finish.py']
    folder=ROOT/'kaggle/e050_submission'
    assert not (ROOT/'results/E050_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,names,'scripts/event_control_runner.py','scripts/weighted_submission_runner.py','biohub-e050-weighted-complement','Biohub E050 Weighted Complement',True,['jarturo/biohub-e041-train']+(['jarturo/biohub-e039-train'] if mode!='dense' else []))
    checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# Weighted complementary tracking\nFull test inference, fixed public tight55, frozen dense detector and reliability-weighted gap completion. No test prediction cache or training labels.']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E050_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(cfg)
if __name__=='__main__':main()
