"""One bounded Kaggle status check; download completed results and advance E034."""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','features','evaluate']); args=parser.parse_args(); stage=args.stage
    api=KaggleApi(); api.authenticate()
    receipt=json.loads((ROOT/f'results/E034_{stage.upper()}_launch.json').read_text())
    status=api.kernels_status(receipt['kernel']); print(stage,status.status.name,flush=True)
    if status.status.name!='COMPLETE': return
    output=ROOT/('outputs/e034_'+stage)
    api.kernels_output(receipt['kernel'],str(output),file_pattern=r'.*(result.json|_metrics.json)$',quiet=True)
    p=output/('temporal_bridge_'+stage)/'result.json'; result=json.loads(p.read_text())
    assert result['status']=='complete' and result['config']==json.loads((ROOT/f'kaggle/temporal_bridge_{stage}/protocol.json').read_text())
    completed=dict(status='complete',kernel=receipt['kernel'],version=receipt['version'],manifest_sha256=sha(p),seconds=result['seconds'])
    (ROOT/f'results/E034_{stage.upper()}_completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    if stage=='evaluate':
        print(json.dumps(result,indent=2)); return
    next_stage='features' if stage=='prepare' else 'evaluate'
    target=ROOT/f'results/E034_{next_stage.upper()}_launch.json'
    if target.exists(): print('Next stage already launched'); return
    if stage=='prepare':
        count=sum(r['donors'] for r in result['videos']); print('Selected donor crops:',count,flush=True)
        if not count: print('No proposals; GPU not launched.'); return
        folder=ROOT/'kaggle/temporal_bridge_features'; meta=json.loads((folder/'kernel-metadata.json').read_text())
        checks=json.loads((ROOT/'results/E034_FEATURES_preflight.json').read_text())
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
        assert sha(folder/'notebook.ipynb')==checks['notebook_sha256']
        response=api.kernels_push(str(folder))
        errors={k:getattr(response,k,None) for k in ['error','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources']}
        assert not any(errors.values()),errors
        launched=dict(kernel=meta['id'],version=response.version_number,url=response.url,launched_at_utc=datetime.now(timezone.utc).isoformat(),notebook_sha256=checks['notebook_sha256'],payload_sha256=checks['payload_sha256'],enable_gpu=True,is_private=True,leaderboard_submitted=False,preparation_manifest_sha256=sha(p))
        target.write_text(json.dumps(launched,indent=2)+'\n'); print(json.dumps(launched),flush=True)
    else:
        subprocess.run([sys.executable,str(ROOT/'scripts/launch_cpu_notebook.py'),str(ROOT/'kaggle/temporal_bridge_evaluate'),'--receipt',str(target)],check=True)

if __name__=='__main__': main()
