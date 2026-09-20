"""Package E031 preparation separately from neural inference."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    config=dict(experiment='E031',videos=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']['calibration'],frames='all',threshold=.2,peak_kernel=3,batch_size=20,modes=['frozen_bn','batch_bn'],gpu_deadline_seconds=1800,training=False,scope='Conditional diagnostic; public detector training membership unknown; no independent holdout claim')
    (ROOT/'baseline/e031_dense_detector.json').write_text(json.dumps(config,indent=2)+'\n')
    folder=ROOT/'kaggle/dense_detector_prepare'
    files=['scripts/dense_detector_prepare_runner.py','baseline/e031_dense_detector.json']
    checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/dense_detector_prepare_runner.py','biohub-dense-detector-prepare-cpu','Biohub Dense Detector Prepare CPU',False,[])
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E031 full calibration images\nCPU preparation, fixed videos, all frames. Public detector provenance unknown. No submission.']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E031_PREP_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(checks)
if __name__=='__main__':main()
