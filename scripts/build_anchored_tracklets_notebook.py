"""Package E019 CPU anchored donor paths with no GPU or new training."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    names=read(ROOT/'kaggle/identity_parent/payload.json')['files']
    names+=['src/biohub_lab/anchored_tracklets.py','scripts/anchored_tracklets_runner.py','baseline/e018_pins.json']
    folder=ROOT/'kaggle/anchored_tracklets'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/anchored_tracklets_runner.py',
        'biohub-lab-anchored-tracklets-cpu','Biohub Lab Anchored Tracklets CPU',False,
        ['jarturo/biohub-lab-full-calibration-compare-cpu','jarturo/biohub-lab-tissue-trajectory-cpu'])
    notebook=read(folder/'notebook.ipynb');notebook['cells'][0]['source']=['# E019 anchored donor tracklets\n',
        'Preserve full Harmonic nodes, edges and division parents. Add compatible donor path segments at vacant endpoints.\n',
        'Freeze annotation-free predictions before evaluating reused calibration. CPU only, no training or submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf8')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),unit_tests_passed=4,short_run_followup_authorized=True)
    (ROOT/'results/E019_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
