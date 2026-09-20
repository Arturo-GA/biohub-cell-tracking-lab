"""Launch E028 neural inference after the paired CPU normalization audit."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
def main():
    root=Path(__file__).resolve().parents[1];target=root/'results/E028_GPU_launch.json'
    if target.exists():raise ValueError('Existing inference launch receipt')
    assert json.loads((root/'results/E028_PREP_completed.json').read_text())['status']=='complete'
    folder=root/'kaggle/cellpose_inference';meta=json.loads((folder/'kernel-metadata.json').read_text())
    checks=json.loads((root/'results/E028_GPU_preflight.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==checks['notebook_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder))
    errors={k:getattr(response,k,None) for k in ['error','invalid_kernel_sources','invalid_dataset_sources','invalid_competition_sources','invalid_model_sources']}
    if any(errors.values()):raise RuntimeError(str(errors))
    record=dict(kernel=meta['id'],version=response.version_number,url=response.url,payload_sha256=checks['payload_sha256'],notebook_sha256=checks['notebook_sha256'],launched_at_utc=datetime.now(timezone.utc).isoformat(),gpu=True,is_private=True,gpu_justification=checks['gpu_justification'],leaderboard_submitted=False)
    target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
