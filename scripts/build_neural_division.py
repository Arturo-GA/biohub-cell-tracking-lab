import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha
def main():
    config=dict(experiment='E037',data_manifest_sha256='03e0f19031df0ed7cb007e1231956e49c38dfaa4fd5cbfe6befed70792cbc8f1',encoder_manifest_sha256='3717afed1078dd01267891ec7676e135613dd43b48a0577c93870ec4e789cc57',old_geometry_ap=.40589285714285717,old_static_ap=.4425595238095238,protocol='Same E035 40 fit/8 development examples and labels. CPU three-time16cubed crops normalized exactly as E033 metadataquantiles and log1p. Two independent frozen E033 encoders; one head per encoder and representation. ExtraTrees256 minleaf3 maxfeatures0.8 seed350920; geometry, neural initial192D, neural sequence576D. Development threshold >=2TP and0FP, unchanged from E035. Additional preregistered neural gate: AP >=old static AP+0.05 and >=same-head shuffled-node AP+0.05. Choose passing representation per encoder on development only. Later graph evaluation uses each encoder/head only on its heldout embryo. No calibration images/labels loaded in preparation or head training. No encoder retraining, no sweep or automatic submission.')
    (ROOT/'baseline/e037_neural_division.json').write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'baseline/e037_graph_pins.json').write_text(json.dumps(dict(features_manifest_sha256='bac1ff43079fb1cb9a6f89d231e00674a0e517056913b0b7bf4254d4aa8deaf6',control_score=.9007529595605399),indent=2)+'\n')
    for stage in ['prepare','features','heads','graph']:
        if (ROOT/f'results/E037_{stage.upper()}_launch.json').exists():continue
        gpu=stage=='features';source=Path('kaggle/temporal_graph_features' if gpu else 'kaggle/identity_parent');runner=f'scripts/neural_division_{stage}_runner.py'
        files=['baseline/e037_neural_division.json','src/biohub_lab/temporal_volume.py','src/biohub_lab/neural_division.py',runner]
        if stage=='graph':files+=['baseline/e037_graph_pins.json','src/biohub_lab/division_sequence.py']
        if not gpu:files+=json.loads((ROOT/source/'payload.json').read_text())['files']
        kernels={'prepare':['jarturo/biohub-division-sequence-prepare-cpu'],'features':['jarturo/biohub-neural-division-prepare-cpu','jarturo/biohub-temporal-volume-train'],'heads':['jarturo/biohub-neural-division-prepare-cpu','jarturo/biohub-neural-division-features'],'graph':['jarturo/biohub-neural-division-heads-cpu','jarturo/biohub-division-sequence-prepare-cpu','jarturo/biohub-temporal-graph-features']}[stage]
        folder=ROOT/f'kaggle/neural_division_{stage}';checks=package(source,folder,files,'scripts/temporal_graph_features_runner.py' if gpu else 'scripts/identity_parent_runner.py',runner,'biohub-neural-division-'+stage+('' if gpu else '-cpu'),'Biohub Neural Division '+stage.title()+('' if gpu else ' CPU'),gpu,kernels)
        nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E037 Neural division representations\n'+config['protocol']];(folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
        (ROOT/f'results/E037_{stage.upper()}_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(stage,checks)
if __name__=='__main__':main()
