"""Freeze CPU confirmation using the exactly equivalent fast visual solver."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    names=read(ROOT/'kaggle/visual_exact/payload.json')['files']+['src/biohub_lab/visual_assignment.py','scripts/visual_validation_runner.py','baseline/e023_validation.json','scripts/visual_submission_postprocess.py']
    folder=ROOT/'kaggle/visual_validation'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/visual_validation_runner.py',
        'biohub-lab-visual-reserved-cpu','Biohub Lab Visual Reserved CPU',False,['jarturo/biohub-lab-visual-reserved-capture'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E023 reserved-video confirmation\n','Frozen visual assignment versus full Harmonic, on eight previously reserved videos.\n','Equivalent fast solver verified on both pilot outputs. Fixed promotion checks; no automatic submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=0,reserved_videos=8,
        exact_pilot_equivalence=read(ROOT/'results/E023_fast_assignment_equivalence.json')['status'])
    (ROOT/'results/E023_CPU_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
