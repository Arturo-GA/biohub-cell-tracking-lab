import json
from pathlib import Path
from build_tissue_trajectory_notebooks import ROOT,package,sha

def main():
    splits=json.loads((ROOT/'baseline/e030_head_protocol.json').read_text())
    config=dict(experiment='E035',**{k:splits[k] for k in ['fit','development','validation']},graph_manifest_sha256='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22',control_score=.9007529595605399,protocol='CPU image measurements, three-frame mother history and three-frame continuity of both daughters. 40 fit/8 development/16 reused calibration videos. Fit/development annotated trajectories jittered1um, all available consecutive contexts; calibration real Harmonic candidates only, known labels require all three GT identities. Up to4 nearest daughters within20um. ExtraTrees256 minleaf3; geometry/static/sequence paired arms. Development gate >=2 true events and zero known false events; threshold chosen on development only. Add missing divisions using orphan daughter(s), preserve existing divisions and nodes. No calibration threshold sweep or automatic submission.')
    (ROOT/'baseline/e035_division_sequence.json').write_text(json.dumps(config,indent=2)+'\n')
    for stage in ['prepare','evaluate']:
        if (ROOT/f'results/E035_{stage.upper()}_launch.json').exists():continue
        files=json.loads((ROOT/'kaggle/identity_parent/payload.json').read_text())['files']+['baseline/e035_division_sequence.json','src/biohub_lab/division_sequence.py',f'scripts/division_sequence_{stage}_runner.py']
        folder=ROOT/f'kaggle/division_sequence_{stage}'
        kernels=['jarturo/biohub-temporal-graph-prepare-cpu'] if stage=='prepare' else ['jarturo/biohub-division-sequence-prepare-cpu']
        checks=package(Path('kaggle/identity_parent'),folder,files,'scripts/identity_parent_runner.py',f'scripts/division_sequence_{stage}_runner.py','biohub-division-sequence-'+stage+'-cpu','Biohub Division Sequence '+stage.title()+' CPU',False,kernels)
        nb=json.loads((folder/'notebook.ipynb').read_text());nb['cells'][0]['source']=['# E035 Division sequences\n'+config['protocol']]
        (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n');checks['notebook_sha256']=sha(folder/'notebook.ipynb')
        (ROOT/f'results/E035_{stage.upper()}_preflight.json').write_text(json.dumps(checks,indent=2)+'\n');print(stage,checks)
if __name__=='__main__':main()
