"""Launch a pinned private GPU payload after its CPU inputs finish."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def main():
    stage=sys.argv[1];folder=ROOT/'kaggle'/stage.lower();receipt=ROOT/f'results/{stage}_launch.json';assert not receipt.exists()
    check=json.loads((ROOT/f'results/{stage}_preflight.json').read_text());meta=json.loads((folder/'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==check['notebook_sha256']
    api=KaggleApi();api.authenticate()
    for source in meta['kernel_sources']:assert api.kernels_status(source).status.name=='COMPLETE',source
    r=api.kernels_push(str(folder));errors={k:getattr(r,k,None) for k in ['error','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources']};assert not any(errors.values()),errors
    result=dict(kernel=r.url.split('/code/',1)[1].strip('/'),requested_kernel=meta['id'],version=r.version_number,url=r.url,launched_at_utc=datetime.now(timezone.utc).isoformat(),**check,leaderboard_submitted=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
