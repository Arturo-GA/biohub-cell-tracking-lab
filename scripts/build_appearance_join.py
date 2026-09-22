import json,sys
from pathlib import Path
from build_three_lines import build
from build_tissue_trajectory_notebooks import package,sha,ROOT
from biohub_lab.appearance_join import ARMS

stage=sys.argv[1]
if stage=='EVALUATE':
    old=json.loads(Path('baseline/e064_protocol.json').read_text())
    cfg=dict(experiment='E066',videos=old['videos'],control_manifest_sha256=old['control_manifest_sha256'],arms=ARMS,method='Native-image appearance and six-frame bidirectional extrapolation to join consecutive existing track ends and starts. Mutual-best ambiguity gate; no node changes or original edge removal.')
    Path('baseline/e066_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e064_evaluate/payload.json').read_text())['files']+['baseline/e066_protocol.json','src/biohub_lab/appearance_join.py','src/biohub_lab/appearance_swap.py']
    build('E066_EVALUATE','scripts/appearance_join_evaluate_runner.py',files,['jarturo/biohub-e061-evaluate-cpu'])
elif stage=='SUBMISSION':
    r=json.loads(Path('results/E066_EVALUATE_completed.json').read_text())['result']
    preview=json.loads(Path('results/E066_PREVIEW_completed.json').read_text())['result']
    assert preview['status']=='complete'
    eligible=[a for a in r['config']['arms'] if preview['reports'][a]['changed_edges']>0]
    assert eligible,'All arms reproduce visible test control; do not spend GPU on a duplicate'
    selected=max(eligible,key=lambda a:r['metrics'][a]['score'])
    r['selected']=selected;r['delta']=r['metrics'][selected]['score']-r['metrics']['control']['score']
    cfg=dict(experiment='E066',selected=r['selected'],swap_options=r['config']['arms'][r['selected']],division_threshold=.25,radius=5.5,visual=False,mode='none',validation_delta=r['delta'],scope='Explicitly requested exploratory new submission. Recompute Harmonic and native-image descriptors on all test images; no cached test predictions or labels.')
    Path('baseline/e066_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e054_control/payload.json').read_text())['files']+['baseline/e066_submission.json','src/biohub_lab/appearance_join.py','src/biohub_lab/appearance_swap.py','scripts/appearance_join_submission_runner.py']
    folder=ROOT/'kaggle/e066_submission';assert not Path('results/E066_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/appearance_join_submission_runner.py','biohub-e066-appearance-join','Biohub E066 Appearance Join',True,[]);checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E066 Native Appearance Tracklet Joins\nFull hidden-test image inference with image-guided additions between track ends and starts.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');Path('results/E066_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
else:raise ValueError(stage)
