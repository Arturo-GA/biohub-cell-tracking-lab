import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    config=dict(experiment='E032',mode='batch_bn',seed=320920,steps=1000,split='Train on one embryo, evaluate the other; both directions',selection='Fixed final step, all represented annotated edges with at least two actual gated parents',train_unknown_near_parent='masked within7um; no background labels',radius_um=20.,candidate_k=12)
    (ROOT/'baseline/e032_temporal_rank.json').write_text(json.dumps(config,indent=2)+'\n')
    files=json.loads((ROOT/'kaggle/identity_parent/payload.json').read_text())['files']+['scripts/dense_temporal_rank_runner.py','baseline/e032_temporal_rank.json']
    folder=ROOT/'kaggle/dense_temporal_rank'
    checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py','scripts/dense_temporal_rank_runner.py','biohub-dense-temporal-rank-cpu','Biohub Dense Temporal Rank CPU',False,['jarturo/biohub-dense-detector-gpu','jarturo/biohub-dense-detector-prepare-cpu'])
    nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E032 actual-candidate temporal ranking\nCPU training on frozen dense-detector features, two embryo directions, image and geometry heads. No encoder training or submission.']
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
    (ROOT/'results/E032_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
if __name__=='__main__':main()
