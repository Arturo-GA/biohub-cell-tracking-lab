"""Evaluate adapted fields with the exact same frozen E024 CPU decoder."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    names=read(ROOT/'kaggle/nucverse_evaluation/payload.json')['files']
    folder=ROOT/'kaggle/point_adapt_evaluation'
    result=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/nucverse_evaluate_runner.py','biohub-lab-point-adapt-evaluation-cpu','Biohub Lab Point Adapt Evaluation CPU',False,['jarturo/biohub-lab-point-adapt-training','jarturo/biohub-lab-nucverse-inputs-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E025 adapted instance segmentation: CPU evaluation\nSame frozen decoder, full frames and reference matching as E024. No neural inference or leaderboard submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    result.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=4,frames=8)
    (ROOT/'results/E025_EVAL_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
