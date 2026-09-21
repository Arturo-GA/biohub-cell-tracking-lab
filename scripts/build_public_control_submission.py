import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha
cfg=dict(experiment='E054_CONTROL',selected='d25_r55',division_threshold=.25,radius=5.5,visual=False,mode='none',reason='E050 pre-complement output exactly matches downloaded public V3 CSV; submit pure ablation with required root filename',expected_visible_csv_sha256='a69c78229c6556d06d5fe9ff050074254b4a326e83249efbf578b843aec7a848')
(ROOT/'baseline/e054_control.json').write_text(json.dumps(cfg,indent=2)+'\n')
files=json.loads((ROOT/'kaggle/e050_submission/payload.json').read_text())['files']
files+=['baseline/e054_control.json','scripts/public_control_submission_runner.py','scripts/joint_submission_runner.py','scripts/joint_submission_finish.py','src/biohub_lab/public_postprocess.py','src/biohub_lab/visual_capture.py']
folder=ROOT/'kaggle/e054_control';assert not (ROOT/'results/E054_CONTROL_launch.json').exists()
checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/public_control_submission_runner.py','biohub-e054-public-control','Biohub E054 Public Control',True,[]);checks.pop('calibration_videos',None)
nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# Pure public tight55 control\nRecompute all test images and export the pure control as submission.csv. No test prediction cache, no training annotations, no complementary bridge stage.']
(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');(ROOT/'results/E054_CONTROL_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(cfg)
