import json,sys
from pathlib import Path
from build_three_lines import build
from build_tissue_trajectory_notebooks import package,sha,ROOT
stage=sys.argv[1]
if stage=='EVALUATE':
    cfg=dict(experiment='E064',videos=json.loads(Path('baseline/e061_protocol.json').read_text())['evaluation'],control_manifest_sha256=json.loads(Path('results/E061_EVALUATE_completed.json').read_text())['manifest_sha256'],arms=dict(image_quarter=dict(weight=.25,temporal=False),temporal_half=dict(weight=.5,temporal=True),temporal_full=dict(weight=1.,temporal=True)),method='Native-image contrast-weighted centroid offsets with Gaussian spatial prior, bounded displacement, temporal median correction, Voronoi ownership and collision/edge guards. Preserve all nodes and links. New localization approach; no training or annotations used to generate predictions.')
    Path('baseline/e064_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e063_evaluate/payload.json').read_text())['files']+['baseline/e064_protocol.json','src/biohub_lab/native_centers.py']
    build('E064_EVALUATE','scripts/native_centers_evaluate_runner.py',files,['jarturo/biohub-e061-evaluate-cpu'])
elif stage=='SUBMISSION':
    r=json.loads(Path('results/E064_EVALUATE_completed.json').read_text())['result'];cfg=dict(experiment='E064',selected=r['selected'],native_options=r['config']['arms'][r['selected']],division_threshold=.25,radius=5.5,visual=False,mode='none',validation_delta=r['delta'],scope='User-authorized exploratory submission. Full Harmonic inference plus native-image coordinate correction, no training labels or cached test predictions.')
    Path('baseline/e064_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e054_control/payload.json').read_text())['files']+['baseline/e064_submission.json','src/biohub_lab/native_centers.py','scripts/native_centers_submission_runner.py']
    folder=ROOT/'kaggle/e064_submission';assert not Path('results/E064_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/native_centers_submission_runner.py','biohub-e064-native-image-centers','Biohub E064 Native Image Centers',True,[]);checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# Native-image temporal center refinement\nFull hidden-test image inference; image-guided coordinate correction. Exploratory submission.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');Path('results/E064_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
else:raise ValueError(stage)
