"""Freeze eight previously reserved videos before reading any of their labels."""
import hashlib,json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    split=read(ROOT/'baseline/event_graph_split.json')['split']
    names=[]
    for prefix in ['44b6_','6bba_']:
        pool=[n for n in split['reserved'] if n.startswith(prefix)]
        names+=sorted(pool,key=lambda n:hashlib.sha256(('biohub-visual-confirmation-v1:'+n).encode()).hexdigest())[:4]
    names=sorted(names);assert len(names)==8 and not set(names)&set(split['calibration']+split['fit']+split['evaluation'])
    config=dict(videos=names,selection='Four filename-hash-selected reserved videos per specimen; no annotation or score selection',
        candidate='E022 visual global assignment',temporal_weight=0.,appearance_probability=.05,capture_topk=8,
        promotion=dict(minimum_score_gain=.005,each_specimen_score_delta_nonnegative=True,improved_videos_at_least_worsened=True,
            division_tp_must_not_decrease=True,division_fp_must_not_increase=True,all_exports_valid=True),
        scope='Previously reserved for this project; public pretrained models may have seen these images. Conditional validation only.')
    path=ROOT/'baseline/e023_validation.json'
    if path.exists():assert read(path)==config
    else:path.write_text(json.dumps(config,indent=2)+'\n')
    files=read(ROOT/'kaggle/visual_capture/payload.json')['files']+['scripts/visual_validation_capture_runner.py','baseline/e023_validation.json']
    folder=ROOT/'kaggle/visual_validation_capture'
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/visual_validation_capture_runner.py',
        'biohub-lab-visual-reserved-capture','Biohub Lab Visual Reserved Capture',True,[])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E023 reserved-video capture\n','Eight filename-selected reserved videos. Frozen exact Harmonic inference plus competing visual links.\n','GPU only for volumetric inference. Scoring and association follow on CPU; no leaderboard submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=0,reserved_videos=8,
        gpu_justification='Exact 3D image inference and TTA on eight reserved videos; CPU handles all association and evaluation')
    (ROOT/'results/E023_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
