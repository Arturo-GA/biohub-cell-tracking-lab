"""Evaluate adapted fields with the exact same frozen E024 CPU decoder."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    names=read(ROOT/'kaggle/nucverse_evaluation/payload.json')['files']
    folder=ROOT/'kaggle/error_window_evaluation'
    result=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/nucverse_evaluate_runner.py','biohub-lab-error-window-evaluation-cpu','Biohub Lab Error Window Evaluation CPU',False,['jarturo/biohub-lab-error-window-inference','jarturo/biohub-lab-error-window-inputs-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E027 adapted instance segmentation: CPU evaluation\nSame frozen decoder as E026. Error-conditioned calibration windows; coverage is a diagnostic, not unbiased accuracy. No neural inference or submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    result.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=16,maximum_frames=48)
    (ROOT/'results/E027_EVAL_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
