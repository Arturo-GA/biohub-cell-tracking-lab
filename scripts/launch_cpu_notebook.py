"""Launch a reviewed private CPU notebook once, without monitoring its run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def validate(folder):
    folder=Path(folder); meta=json.loads((folder/'kernel-metadata.json').read_text())
    if meta.get('is_private') is not True or meta.get('enable_gpu') is not False or meta.get('enable_tpu') is not False:
        raise ValueError('Only explicitly private CPU notebooks are permitted')
    if meta.get('machine_shape'):
        raise ValueError('Remove any accelerator machine shape from this CPU workflow')
    return meta


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('--receipt',required=True)
    args=parser.parse_args();folder=Path(args.folder);receipt=Path(args.receipt)
    if receipt.exists():raise ValueError('Launch receipt exists; refuse an accidental duplicate')
    meta=validate(folder)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder))
    errors={k:getattr(response,k,None) for k in ('error','invalid_dataset_sources','invalid_competition_sources','invalid_kernel_sources','invalid_model_sources')}
    errors={k:v for k,v in errors.items() if v}
    if errors:raise RuntimeError(json.dumps(errors))
    result=dict(kernel=meta['id'],version=getattr(response,'version_number',None),kernel_id=getattr(response,'kernel_id',None),
        url=getattr(response,'url',None),ref=getattr(response,'ref',None),launched_at_utc=datetime.now(timezone.utc).isoformat(),
        payload_sha256=json.loads((folder/'payload.json').read_text())['sha256'],
        notebook_sha256=hashlib.sha256((folder/meta['code_file']).read_bytes()).hexdigest(),
        enable_gpu=False,enable_tpu=False,is_private=True,monitoring=False,leaderboard_submitted=False)
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
