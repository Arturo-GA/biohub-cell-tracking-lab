"""Two-video image-inference smoke; no GPU evaluation or training."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    calibration=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    names=[next(n for n in calibration if n.startswith(prefix)) for prefix in ['44b6_','6bba_']]
    (ROOT/'baseline/e022_capture.json').write_text(json.dumps(dict(videos=names,selection='First sorted calibration filename per specimen, independent of results',purpose='Exact alternative graph capture and runtime smoke'),indent=2)+'\n')
    files=read(ROOT/'kaggle/calibration_control/payload.json')['files']+['src/biohub_lab/visual_capture.py','scripts/visual_capture_runner.py','baseline/e022_capture.json']
    folder=ROOT/'kaggle/visual_capture'
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/visual_capture_runner.py',
        'biohub-lab-visual-candidate-capture','Biohub Lab Visual Candidate Capture',True,[])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E022 exact visual candidate capture\n','Two fixed videos, unchanged Harmonic predictions plus alternative-edge evidence.\n','GPU only for dense 3D inference. Evaluation and optimization follow on CPU. No submission.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=2,gpu_justification='Dense 3D image inference and TTA; candidate graph optimization/evaluation remain CPU')
    (ROOT/'results/E022_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
