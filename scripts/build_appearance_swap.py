import json,sys
from pathlib import Path
from build_three_lines import build
from build_tissue_trajectory_notebooks import package,sha,ROOT
from biohub_lab.appearance_swap import ARMS

stage=sys.argv[1]
if stage=='EVALUATE':
    old=json.loads(Path('baseline/e064_protocol.json').read_text())
    cfg=dict(experiment='E065',videos=old['videos'],control_manifest_sha256=old['control_manifest_sha256'],arms=ARMS,method='Native-image four-frame appearance descriptors plus bidirectional motion gates; disjoint degree-preserving 2-opt swaps of existing non-division links. No node changes, no training or labels.')
    Path('baseline/e065_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e064_evaluate/payload.json').read_text())['files']+['baseline/e065_protocol.json','src/biohub_lab/appearance_swap.py']
    build('E065_EVALUATE','scripts/appearance_swap_evaluate_runner.py',files,['jarturo/biohub-e061-evaluate-cpu'])
elif stage=='SUBMISSION':
    r=json.loads(Path('results/E065_EVALUATE_completed.json').read_text())['result']
    preview=json.loads(Path('results/E065_PREVIEW_completed.json').read_text())['result']
    assert preview['status']=='complete' and preview['reports'][r['selected']]['changed_edges']>0,'Selected arm would reproduce visible test control; do not spend GPU on a duplicate'
    cfg=dict(experiment='E065',selected=r['selected'],swap_options=r['config']['arms'][r['selected']],division_threshold=.25,radius=5.5,visual=False,mode='none',validation_delta=r['delta'],scope='Explicitly requested exploratory new submission. Recompute Harmonic and native-image descriptors on all test images; no cached test predictions or labels.')
    Path('baseline/e065_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e054_control/payload.json').read_text())['files']+['baseline/e065_submission.json','src/biohub_lab/appearance_swap.py','scripts/appearance_swap_submission_runner.py']
    folder=ROOT/'kaggle/e065_submission';assert not Path('results/E065_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/appearance_swap_submission_runner.py','biohub-e065-appearance-swap','Biohub E065 Appearance Swap',True,[]);checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E065 Native Appearance Tracklet Swaps\nFull hidden-test image inference with image-guided degree-preserving reassociation.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');Path('results/E065_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
else:raise ValueError(stage)
