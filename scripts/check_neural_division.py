import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    stage=sys.argv[1];assert stage in ['prepare','features','heads','graph']
    receipt=json.loads((ROOT/f'results/E037_{stage.upper()}_launch.json').read_text());api=KaggleApi();api.authenticate();status=api.kernels_status(receipt['kernel']);print(stage,status.status.name,flush=True)
    if status.status.name!='COMPLETE':return
    output=ROOT/f'outputs/e037_{stage}'
    api.kernels_output(receipt['kernel'],str(output),file_pattern=r'.*(result.json|_model.pkl|_metrics.json|_development.npz)$',quiet=True)
    folder={'prepare':'neural_division_data','features':'neural_division_features','heads':'neural_division_heads','graph':'neural_division_graph'}[stage]
    p=output/folder/'result.json';result=json.loads(p.read_text());assert result['status']=='complete' and result['config']==json.loads((ROOT/'baseline/e037_neural_division.json').read_text())
    (ROOT/f'results/E037_{stage.upper()}_completed.json').write_text(json.dumps(dict(status='complete',version=receipt['version'],kernel=receipt['kernel'],manifest_sha256=sha(p),seconds=result['seconds']),indent=2)+'\n')
    if stage=='heads':
        print(json.dumps({k:result[k] for k in ['heads','choices','proceed_to_graph']},indent=2))
        if not result['proceed_to_graph']:return
    if stage=='graph':print(json.dumps(result,indent=2));return
    next_stage={'prepare':'features','features':'heads','heads':'graph'}[stage];target=ROOT/f'results/E037_{next_stage.upper()}_launch.json'
    if target.exists():print('Next stage already launched');return
    if stage=='prepare':
        count=sum(r['nodes'] for r in result['videos']);assert count>0;print('Unique crops:',count,flush=True)
        folder=ROOT/'kaggle/neural_division_features';meta=json.loads((folder/'kernel-metadata.json').read_text());check=json.loads((ROOT/'results/E037_FEATURES_preflight.json').read_text())
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu'];assert sha(folder/'notebook.ipynb')==check['notebook_sha256']
        response=api.kernels_push(str(folder));errors={k:getattr(response,k,None) for k in ['error','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources']};assert not any(errors.values()),errors
        launched=dict(kernel=meta['id'],version=response.version_number,url=response.url,launched_at_utc=datetime.now(timezone.utc).isoformat(),payload_sha256=check['payload_sha256'],notebook_sha256=check['notebook_sha256'],is_private=True,enable_gpu=True,preparation_manifest_sha256=sha(p),leaderboard_submitted=False)
        target.write_text(json.dumps(launched,indent=2)+'\n');print(json.dumps(launched),flush=True)
    else:
        folder=ROOT/f'kaggle/neural_division_{next_stage}'
        if not folder.exists():print('Passing heads ready; graph stage must be packaged before launch.');return
        subprocess.run([sys.executable,str(ROOT/'scripts/launch_cpu_notebook.py'),str(folder),'--receipt',str(target)],check=True)
if __name__=='__main__':main()
