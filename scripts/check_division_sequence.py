import hashlib,json,subprocess,sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def main():
    stage=sys.argv[1];assert stage in ['prepare','evaluate']
    receipt=json.loads((ROOT/f'results/E035_{stage.upper()}_launch.json').read_text())
    api=KaggleApi();api.authenticate();s=api.kernels_status(receipt['kernel']);print(stage,s.status.name,flush=True)
    if s.status.name!='COMPLETE':return
    output=ROOT/('outputs/e035_'+stage);api.kernels_output(receipt['kernel'],str(output),file_pattern=r'.*(result.json|_metrics.json|_model.pkl)$',quiet=True)
    folder='division_sequence_data' if stage=='prepare' else 'division_sequence_evaluation';p=output/folder/'result.json';r=json.loads(p.read_text());assert r['status']=='complete'
    assert r['config']==json.loads((ROOT/'baseline/e035_division_sequence.json').read_text())
    saved=dict(status='complete',kernel=receipt['kernel'],version=receipt['version'],manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=r['seconds'])
    (ROOT/f'results/E035_{stage.upper()}_completed.json').write_text(json.dumps(saved,indent=2)+'\n')
    if stage=='prepare':
        for split in ['fit','development','validation']:
            rows=[v for v in r['videos'] if v['split']==split];print(split,{k:sum(v[k] for v in rows) for k in ['candidates','positive','negative','unknown','annotated_divisions']},flush=True)
        target=ROOT/'results/E035_EVALUATE_launch.json'
        if not target.exists():subprocess.run([sys.executable,str(ROOT/'scripts/launch_cpu_notebook.py'),str(ROOT/'kaggle/division_sequence_evaluate'),'--receipt',str(target)],check=True)
    else:print(json.dumps({k:r[k] for k in ['heads','metrics','seconds']},indent=2))
if __name__=='__main__':main()
