"""Prepare an inference-only notebook; launch only after reserved validation."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,sha,package
def main():
    names=['NOTICE.md','baseline/harmonic_inference.py','src/biohub_lab/__init__.py',
        'src/biohub_lab/patch.py','src/biohub_lab/visual_capture.py','src/biohub_lab/visual_candidates.py',
        'src/biohub_lab/visual_assignment.py','src/biohub_lab/detection_identity.py','src/biohub_lab/association_rank.py',
        'src/biohub_lab/submission.py','src/biohub_lab/calibration_export.py',
        'scripts/visual_submission_runner.py','scripts/visual_submission_postprocess.py']
    folder=ROOT/'kaggle/visual_submission'
    checks=package(Path('kaggle/event_graph_control'),folder,names,'scripts/event_control_runner.py','scripts/visual_submission_runner.py',
        'biohub-lab-visual-global-assignment','Biohub Lab Visual Global Assignment',True,[])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# Biohub visual global assignment\n','Frozen Harmonic image inference, retaining competing learned links before pruning. Global visual assignment preserves detections and division edges.\n','Offline test inference only. No training labels, cached train predictions, or dataset-specific policies.\n']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    checks.update(notebook_sha256=sha(folder/'notebook.ipynb'),calibration_videos=0,test_inference=True,
        leaderboard_submitted=False,launch_pending_reserved_validation=True,contains_training_predictions=False,
        gpu_justification='Full volumetric neural inference; following association uses CPU with the equivalent fast solver')
    (ROOT/'results/VISUAL_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':main()
