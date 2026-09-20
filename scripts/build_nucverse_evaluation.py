"""CPU decoding and metric preparation, with no TensorFlow dependency."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    names=read(ROOT/'kaggle/identity_parent/payload.json')['files']+['vendor/nucverse/LICENSE','NOTICE.md','src/biohub_lab/nucverse_instances.py','scripts/nucverse_evaluate_runner.py']
    folder=ROOT/'kaggle/nucverse_evaluation'
    result=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/nucverse_evaluate_runner.py','biohub-lab-nucverse-evaluation-cpu','Biohub Lab NucVerse Evaluation CPU',False,['jarturo/biohub-lab-nucverse-inference','jarturo/biohub-lab-nucverse-inputs-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E024 CPU instance decoding and coverage\nNo neural inference or training. Sparse unmatched predictions are not false-positive labels.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    result.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=4,frames=8)
    (ROOT/'results/E024_EVAL_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
