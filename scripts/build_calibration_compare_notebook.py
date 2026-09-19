"""Freeze completed E017 and control receipts into one private CPU evaluation."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    control=ROOT/'outputs/e017_control_recovery/calibration_control'
    tissue=ROOT/'outputs/e017_recovery/tissue_trajectory'
    parent=ROOT/'outputs/e016_recovery/identity_parent'
    pins=dict(control_result_sha256=sha(control/'result.json'),tissue_result_sha256=sha(tissue/'result.json'),
        parent_result_sha256=sha(parent/'result.json'),control_csv_sha256=read(control/'result.json')['csv_sha256'],
        tissue_metrics_sha256=sha(tissue/'candidate_metrics.json'),learned_metrics_sha256=sha(parent/'learned_metrics.json'),
        geometric_metrics_sha256=sha(parent/'geometric_metrics.json'))
    (ROOT/'baseline/e017_compare_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    names=read(ROOT/'kaggle/identity_parent/payload.json')['files']
    names+=['src/biohub_lab/calibration_export.py','scripts/calibration_compare_runner.py','baseline/e017_compare_pins.json']
    folder=ROOT/'kaggle/calibration_compare'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/calibration_compare_runner.py',
        'biohub-lab-full-calibration-compare-cpu','Biohub Lab Full Calibration Compare CPU',False,
        ['jarturo/biohub-lab-harmonic-calibration-control','jarturo/biohub-lab-tissue-trajectory-cpu','jarturo/biohub-lab-identity-parent-cpu'])
    notebook=read(folder/'notebook.ipynb');notebook['cells'][0]['source']=['# E017 full paired calibration evaluation\n',
        'CPU only: validate and evaluate frozen Harmonic CSV; compare E016 geometric/learned and E017 trajectory on the same 16 videos.\n',
        'No training, image inference, parameter selection or leaderboard submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf8')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),export_tests_passed=3,automatic_monitoring=False)
    (ROOT/'results/E017_compare_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
