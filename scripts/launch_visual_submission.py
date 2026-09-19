"""Launch the prepared private inference notebook only after frozen checks pass."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1];target=root/'results/VISUAL_SUBMISSION_launch.json'
    if target.exists():raise ValueError('Existing launch receipt; refuse duplicate execution')
    validation=json.loads((root/'results/E023_completed.json').read_text())
    assert validation['status']=='complete' and validation['promote']
    assert validation['production_csv_exactly_matches_frozen'] and all(validation['promotion_checks'].values())
    folder=root/'kaggle/visual_submission';metadata=json.loads((folder/'kernel-metadata.json').read_text())
    preflight=json.loads((root/'results/VISUAL_SUBMISSION_preflight.json').read_text())
    payload=json.loads((folder/'payload.json').read_text())
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet'] and not metadata['enable_tpu']
    assert not metadata['kernel_sources']
    assert payload['sha256']==preflight['payload_sha256']
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==preflight['notebook_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder))
    errors={k:getattr(response,k,None) for k in ['error','invalid_dataset_sources','invalid_competition_sources','invalid_kernel_sources','invalid_model_sources']}
    if any(errors.values()):raise RuntimeError(str(errors))
    record=dict(kernel=metadata['id'],version=response.version_number,url=response.url,
        payload_sha256=payload['sha256'],notebook_sha256=preflight['notebook_sha256'],
        launched_at_utc=datetime.now(timezone.utc).isoformat(),enable_gpu=True,is_private=True,
        reserved_validation_receipt_sha256=hashlib.sha256((root/'results/E023_completed.json').read_bytes()).hexdigest(),
        leaderboard_submitted=False,gpu_justification=preflight['gpu_justification'])
    target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
