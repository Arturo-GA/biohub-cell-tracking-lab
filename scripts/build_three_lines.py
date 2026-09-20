"""Versioned private packages for the three user-authorized research lines."""
import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def build(stage,runner,files,kernels,gpu=False,internet=False):
    receipt=ROOT/f'results/{stage}_launch.json'
    if receipt.exists():raise RuntimeError('Already launched: '+stage)
    source=Path('kaggle/temporal_graph_features' if gpu else 'kaggle/identity_parent')
    if not gpu:files+=json.loads((ROOT/source/'payload.json').read_text())['files']
    slug='biohub-'+stage.lower().replace('_','-')+('' if gpu else '-cpu')
    folder=ROOT/'kaggle'/stage.lower()
    checks=package(source,folder,files+[runner],'scripts/temporal_graph_features_runner.py' if gpu else 'scripts/identity_parent_runner.py',runner,slug,'Biohub '+stage.replace('_',' ')+('' if gpu else ' CPU'),gpu,kernels)
    checks.pop('calibration_videos',None)  # The source wrapper used an unrelated fixed16 cohort.
    meta=json.loads((folder/'kernel-metadata.json').read_text());meta['enable_internet']=internet;(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# '+stage+'\nSee packaged protocol. No automatic leaderboard submission.'];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/f'results/{stage}_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(stage,checks)
if __name__=='__main__':
    build('E038_AUDIT','scripts/three_lines_audit_runner.py',['baseline/e035_division_sequence.json','src/biohub_lab/temporal_data.py'],[],internet=True)
