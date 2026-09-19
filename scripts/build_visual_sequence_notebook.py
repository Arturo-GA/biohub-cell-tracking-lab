"""Build E020 using existing GPU evidence on CPU."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    names=read(ROOT/'kaggle/identity_parent/payload.json')['files']
    names+=['src/biohub_lab/visual_sequence.py','scripts/visual_sequence_runner.py','baseline/e018_pins.json']
    folder=ROOT/'kaggle/visual_sequence'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/visual_sequence_runner.py',
        'biohub-lab-visual-sequence-cpu','Biohub Lab Visual Sequence CPU',False,
        ['jarturo/biohub-lab-full-calibration-compare-cpu','jarturo/biohub-lab-harmonic-calibration-control'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E020 Visual sequence assignment\n','Reuse dense learned probabilities. Preserve detections and division edges. Compare visual-only and second-order trajectory energy.\n','CPU only. Both prediction arms frozen before evaluation.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),short_run_followup_authorized=True)
    (ROOT/'results/E020_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
