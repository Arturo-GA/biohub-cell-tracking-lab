"""Prepare a fixed non-overlapping fit/calibration domain adaptation experiment."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    split=read(ROOT/'baseline/event_graph_split.json')['split']
    config=dict(fit=split['fit'],validation=split['calibration'],patch=[32,64,64],seed=250920,
        frames_per_fit_video=8,crops_per_fit_frame=4,frames_per_validation_video=4,
        steps=800,learning_rate=1e-5,validate_every=200,gpu_training_deadline_seconds=3600,
        weights=dict(positive=5.,dark=1.,teacher_mask=.1,flow=.5,teacher_flow=.01),
        selection='Fixed final step; held-out losses are diagnostics, no threshold sweep or leaderboard selection',
        source='NucVerse generalist E024; real Biohub fit images and partial annotation cores')
    (ROOT/'baseline/e025_point_adapt.json').write_text(json.dumps(config,indent=2)+'\n')
    files=read(ROOT/'kaggle/identity_parent/payload.json')['files']+['scripts/point_adapt_data_runner.py','src/biohub_lab/point_supervision.py','baseline/e025_point_adapt.json']
    folder=ROOT/'kaggle/point_adapt_data'
    checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/point_adapt_data_runner.py','biohub-lab-point-adapt-data-cpu','Biohub Lab Point Adapt Data CPU',False,[])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E025 partial point-supervised domain adaptation data\n48 fit videos; 16 separate calibration videos. CPU only. Real images, annotated positive cores, photometric weak negatives, unknown elsewhere.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),fit_videos=len(config['fit']),calibration_videos=len(config['validation']))
    (ROOT/'results/E025_DATA_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
