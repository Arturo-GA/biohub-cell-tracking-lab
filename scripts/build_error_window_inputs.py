"""Freeze one error-centered full-image window per calibration video."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,read,package,sha
def main():
    config=dict(videos=read(ROOT/'baseline/event_graph_split.json')['split']['calibration'],control_sha256=read(ROOT/'baseline/e018_pins.json')['harmonic_csv_sha256'],selection='One anchor per calibration video with unmatched annotated centers; prioritize missing division relatives, then missing count, then earliest frame; full frames t-1,t,t+1',annotations_used_to_select_frames=True,patch=[64,128,128],stride=[48,96,96],gpu_deadline_seconds=1800,preprocessing='Same as E026',weight_source='Frozen E025 checkpoint; E026 image-local normalization',scope='Error-conditioned diagnostic, never an unbiased tracking or leaderboard estimate')
    (ROOT/'baseline/e027_error_windows.json').write_text(json.dumps(config,indent=2)+'\n')
    files=read(ROOT/'kaggle/visual_validation/payload.json')['files']+['scripts/error_window_inputs_runner.py','src/biohub_lab/nucverse_instances.py','baseline/e027_error_windows.json']
    folder=ROOT/'kaggle/error_window_inputs'
    result=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/error_window_inputs_runner.py','biohub-lab-error-window-inputs-cpu','Biohub Lab Error Window Inputs CPU',False,['jarturo/biohub-lab-full-calibration-compare-cpu'])
    nb=read(folder/'notebook.ipynb');nb['cells'][0]['source']=['# E027 control-error windows\nCPU preparation. Labels select diagnostic windows; inference receives full images. Explicitly error-conditioned calibration, no submission or training.\n'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');result['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E027_INPUTS_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
