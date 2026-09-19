"""Build E021 pretrained attention replay with all candidate alternatives."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    source=ROOT/'outputs/e017_control_recovery/calibration_control/harmonic_control/tracking_repo/src/biohub_tracking/models/simple_node_transformer.py'
    pins=dict(source_sha256=sha(source),checkpoint_sha256='9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f')
    (ROOT/'baseline/e021_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    names=read(ROOT/'kaggle/visual_sequence/payload.json')['files']+['src/biohub_lab/visual_replay.py','scripts/visual_replay_runner.py','baseline/e021_pins.json']
    folder=ROOT/'kaggle/visual_replay'
    checks=package(Path('kaggle/identity_parent'),folder,names,'scripts/identity_parent_runner.py','scripts/visual_replay_runner.py',
        'biohub-lab-visual-replay-cpu','Biohub Lab Visual Replay CPU',False,
        ['jarturo/biohub-lab-full-calibration-compare-cpu','jarturo/biohub-lab-harmonic-calibration-control','jarturo/biohub-lab-learned-event-graph'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E021 Pretrained visual attention replay\n','Recover competing associations with a frozen public Transformer, using cached image features. Then optimize whole trajectories.\n','Approximate feature replay, not exact Harmonic reproduction. CPU only. Fixed 16-video reused calibration.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),annotations_used_for_predictions=False)
    (ROOT/'results/E021_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
