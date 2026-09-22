import json,sys
from pathlib import Path
from build_three_lines import build
from build_tissue_trajectory_notebooks import package,sha,ROOT
from biohub_lab.trajectory_denoise import ARMS

stage=sys.argv[1]
if stage=='EVALUATE':
    old=json.loads(Path('baseline/e064_protocol.json').read_text())
    cfg=dict(experiment='E067',videos=old['videos'],control_manifest_sha256=old['control_manifest_sha256'],arms=ARMS,method='Robust whole-track penalized acceleration denoising. Four IRLS iterations, no crossing divisions, cap 1.2 um before native rounding and identity safeguards. No labels or node/edge changes.')
    Path('baseline/e067_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e064_evaluate/payload.json').read_text())['files']+['baseline/e067_protocol.json','src/biohub_lab/trajectory_denoise.py']
    build('E067_EVALUATE','scripts/trajectory_denoise_evaluate_runner.py',files,['jarturo/biohub-e061-evaluate-cpu'])
elif stage=='SUBMISSION':
    r=json.loads(Path('results/E067_EVALUATE_completed.json').read_text())['result'];preview=json.loads(Path('results/E067_PREVIEW_completed.json').read_text())
    eligible=[a for a in r['config']['arms'] if preview['reports'][a]['moved']>0];assert eligible
    selected=max(eligible,key=lambda a:r['metrics'][a]['score'])
    other=json.loads(Path('results/E066_EVALUATE_completed.json').read_text())['result']
    assert r['metrics'][selected]['score']>=max(x['score'] for x in other['metrics'].values()),'E066 is better on the common validation; defer this GPU run for final selection'
    cfg=dict(experiment='E067',selected=selected,denoise_options=r['config']['arms'][selected],division_threshold=.25,radius=5.5,visual=False,mode='none',validation_delta=r['metrics'][selected]['score']-r['metrics']['control']['score'],scope='Explicitly requested exploratory submission. Full Harmonic test inference followed by robust full-track denoising, no labels or cached test predictions.')
    Path('baseline/e067_submission.json').write_text(json.dumps(cfg,indent=2)+'\n')
    files=json.loads(Path('kaggle/e054_control/payload.json').read_text())['files']+['baseline/e067_submission.json','src/biohub_lab/trajectory_denoise.py','src/biohub_lab/native_centers.py','scripts/trajectory_denoise_submission_runner.py']
    folder=ROOT/'kaggle/e067_submission';assert not Path('results/E067_SUBMISSION_launch.json').exists()
    checks=package(Path('kaggle/event_graph_control'),folder,files,'scripts/event_control_runner.py','scripts/trajectory_denoise_submission_runner.py','biohub-e067-trajectory-denoise','Biohub E067 Trajectory Denoise',True,[]);checks.pop('calibration_videos',None)
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E067 Robust Whole-Track Denoising\nFull hidden-test inference; robust penalized acceleration correction with fixed graph.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb');Path('results/E067_SUBMISSION_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
else:raise ValueError(stage)
