"""Package CPU comparison for the exact captured graph."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    names=read(ROOT/'kaggle/visual_sequence/payload.json')['files']+['src/biohub_lab/visual_candidates.py','src/biohub_lab/calibration_export.py','scripts/visual_exact_runner.py','baseline/e022_capture.json']
    folder=ROOT/'kaggle/visual_exact'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/visual_exact_runner.py',
        'biohub-lab-visual-exact-cpu','Biohub Lab Visual Exact CPU',False,['jarturo/biohub-lab-visual-candidate-capture'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E022 exact visual graph CPU comparison\n','Unchanged full Harmonic versus global visual assignment and sequence energy, on two fixed videos.\n','All predictions frozen before evaluation. No training, GPU, or leaderboard submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=2)
    (ROOT/'results/E022_CPU_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
