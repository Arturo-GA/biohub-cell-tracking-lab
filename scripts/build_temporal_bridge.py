import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT, package, sha

def main():
    base=dict(experiment='E034',graph_manifest_sha256='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22',dense_manifest_sha256='a09a6f6533ff80f0b6e202fb2b3314683dda7cd29029381853c67e030671a149',training_manifest_sha256='3717afed1078dd01267891ec7676e135613dd43b48a0577c93870ec4e789cc57',harmonic_features_sha256='bac1ff43079fb1cb9a6f89d231e00674a0e517056913b0b7bf4254d4aa8deaf6',control_score=.9007529595605399,protocol='Track ends to starts dt2..4; dense donors >3um from existing nodes, <=4um from interpolation, second donor distance margin>=1um, max8um/frame; reject competing endpoints, prohibit donor reuse. Image arm min adjacent cosine>=0.8, cross-embryo encoder. Fixed protocol, no sweep. Reused16-video calibration, upstream detector split unknown.')
    for stage in ['prepare','features','evaluate']:
        if (ROOT/f'results/E034_{stage.upper()}_launch.json').exists(): continue
        config=dict(base,stage=stage); (ROOT/'baseline/e034_bridge.json').write_text(json.dumps(config,indent=2)+'\n')
        gpu=stage=='features'; source=Path('kaggle/temporal_graph_features' if gpu else 'kaggle/identity_parent')
        files=['baseline/e034_bridge.json','scripts/temporal_bridge_runner.py','src/biohub_lab/temporal_bridge.py','src/biohub_lab/temporal_volume.py']
        if not gpu: files+=json.loads((ROOT/source/'payload.json').read_text())['files']
        kernels=['jarturo/biohub-temporal-graph-prepare-cpu']
        if stage=='prepare': kernels+=['jarturo/biohub-dense-detector-gpu']
        else: kernels+=['jarturo/biohub-temporal-bridge-prepare-cpu']
        if gpu: kernels+=['jarturo/biohub-temporal-volume-train']
        if stage=='evaluate': kernels+=['jarturo/biohub-temporal-bridge-features','jarturo/biohub-temporal-graph-features']
        folder=ROOT/('kaggle/temporal_bridge_'+stage)
        checks=package(source,folder,files,'scripts/temporal_graph_features_runner.py' if gpu else 'scripts/identity_parent_runner.py','scripts/temporal_bridge_runner.py','biohub-temporal-bridge-'+stage+('' if gpu else '-cpu'),'Biohub Temporal Bridge '+stage.title()+('' if gpu else ' CPU'),gpu,kernels)
        nb=json.loads((folder/'notebook.ipynb').read_text()); nb['cells'][0]['source']=['# E034 Temporal bridge '+stage+'\n'+base['protocol']]
        (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n'); checks['notebook_sha256']=sha(folder/'notebook.ipynb')
        (ROOT/f'results/E034_{stage.upper()}_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        (folder/'protocol.json').write_text(json.dumps(config,indent=2)+'\n'); print(stage,checks)

if __name__=='__main__': main()
