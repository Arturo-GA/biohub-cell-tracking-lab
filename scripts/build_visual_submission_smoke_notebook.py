"""CPU verification of the exact production path before submission launch."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    files=read(ROOT/'kaggle/visual_validation/payload.json')['files']+['scripts/visual_submission_postprocess.py','scripts/visual_submission_smoke_runner.py']
    folder=ROOT/'kaggle/visual_submission_smoke'
    checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/visual_submission_smoke_runner.py',
        'biohub-lab-visual-production-check','Biohub Lab Visual Production Check',False,
        ['jarturo/biohub-lab-visual-candidate-capture','jarturo/biohub-lab-visual-exact-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# Production postprocessing check\n','CPU only. Run the final inference postprocessor on both real pilot outputs and require the exact prior CSV hash.\n','No new image inference, labels, training, or submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=2)
    (ROOT/'results/VISUAL_PRODUCTION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
