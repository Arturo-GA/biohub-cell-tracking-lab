"""Launch E031 once after successful CPU preparation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1];receipt=root/'results/E031_GPU_launch.json'
    assert not receipt.exists(),'Already launched'
    prep=json.loads((root/'outputs/e031_prepare/dense_detector_inputs/result.json').read_text())
    config=json.loads((root/'baseline/e031_dense_detector.json').read_text())
    assert prep['status']=='complete' and prep['config']==config and len(prep['videos'])==16
    folder=root/'kaggle/dense_detector_gpu';checks=json.loads((root/'results/E031_GPU_preflight.json').read_text());meta=json.loads((folder/'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==checks['notebook_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    assert api.kernels_status('jarturo/biohub-dense-detector-prepare-cpu').status.name=='COMPLETE'
    response=api.kernels_push(str(folder))
    errors={k:getattr(response,k,None) for k in ['error','invalid_kernel_sources','invalid_dataset_sources','invalid_competition_sources','invalid_model_sources']}
    assert not any(errors.values()),errors
    result=dict(kernel=meta['id'],url=response.url,version=response.version_number,launched_at_utc=datetime.now(timezone.utc).isoformat(),**checks,gpu=True,is_private=True,leaderboard_submitted=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
