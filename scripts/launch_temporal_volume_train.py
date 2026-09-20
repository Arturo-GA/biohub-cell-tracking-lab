import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1];receipt=root/'results/E033_TRAIN_launch.json';assert not receipt.exists()
    folder=root/'kaggle/temporal_volume_train';meta=json.loads((folder/'kernel-metadata.json').read_text());checks=json.loads((root/'results/E033_TRAIN_preflight.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==checks['notebook_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();assert api.kernels_status('jarturo/biohub-temporal-volume-prepare-cpu').status.name=='COMPLETE'
    response=api.kernels_push(str(folder));errors={k:getattr(response,k,None) for k in ['error','invalid_kernel_sources','invalid_dataset_sources','invalid_competition_sources','invalid_model_sources']};assert not any(errors.values()),errors
    record=dict(kernel=meta['id'],url=response.url,version=response.version_number,launched_at_utc=datetime.now(timezone.utc).isoformat(),**checks,training=True,leaderboard_submitted=False)
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
