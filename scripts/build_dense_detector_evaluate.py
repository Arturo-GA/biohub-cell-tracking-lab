import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    protocol=dict(control_sha256=json.loads((ROOT/'baseline/e018_pins.json').read_text())['harmonic_csv_sha256'],radius_um=14.,minimum_track_length=6)
    (ROOT/'baseline/e031_evaluation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    files=json.loads((ROOT/'kaggle/visual_validation/payload.json').read_text())['files']+['scripts/dense_detector_evaluate_runner.py','src/biohub_lab/nucverse_instances.py','baseline/e031_dense_detector.json','baseline/e031_evaluation_protocol.json']
    folder=ROOT/'kaggle/dense_detector_evaluate'
    checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/dense_detector_evaluate_runner.py','biohub-dense-detector-evaluate-cpu','Biohub Dense Detector Evaluate CPU',False,['jarturo/biohub-dense-detector-gpu','jarturo/biohub-dense-detector-prepare-cpu','jarturo/biohub-lab-full-calibration-compare-cpu'])
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E031 complete CPU comparison\nCoverage, equal-count diagnostic, geometric tracking and official metric on 16 complete calibration videos. Public detector training membership unknown. No submission.']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E031_EVAL_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
if __name__=='__main__':main()
