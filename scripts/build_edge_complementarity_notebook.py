"""Package E018 official-matcher complementarity diagnostic, private CPU."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    compare=ROOT/'outputs/e017_compare_recovery/calibration_compare';tissue=ROOT/'outputs/e017_recovery/tissue_trajectory'
    pins=dict(compare_result_sha256=sha(compare/'result.json'),tissue_result_sha256=sha(tissue/'result.json'),
        harmonic_csv_sha256=read(compare/'export_receipt.json')['validated_sha256'],
        tissue_csv_sha256=read(tissue/'frozen_predictions.json')['csv_sha256'],
        harmonic_metrics_sha256=sha(compare/'harmonic_metrics.json'),tissue_metrics_sha256=sha(tissue/'candidate_metrics.json'))
    (ROOT/'baseline/e018_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    names=read(ROOT/'kaggle/identity_parent/payload.json')['files']
    names+=['src/biohub_lab/edge_complementarity.py','scripts/edge_complementarity_runner.py','baseline/e018_pins.json']
    folder=ROOT/'kaggle/edge_complementarity'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/edge_complementarity_runner.py',
        'biohub-lab-edge-complementarity-cpu','Biohub Lab Edge Complementarity CPU',False,
        ['jarturo/biohub-lab-full-calibration-compare-cpu','jarturo/biohub-lab-tissue-trajectory-cpu'])
    notebook=read(folder/'notebook.ipynb');notebook['cells'][0]['source']=['# E018 official-matcher edge complementarity\n',
        'CPU-only diagnosis of correct links unique to Harmonic or E017. Reuses frozen predictions, reproduces official TP/FP counts.\n',
        'Ground-truth overlap is diagnostic only: no fusion, training, threshold selection or submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf8')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),unit_tests_passed=4,automatic_monitoring=False)
    (ROOT/'results/E018_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
