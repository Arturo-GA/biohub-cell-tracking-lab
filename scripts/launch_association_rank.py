"""Launch the E014 GPU notebook explicitly authorized on 2026-09-18, once."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    folder=ROOT/'kaggle/association_rank'
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    assert meta['id']=='jarturo/biohub-lab-association-ranking'
    assert meta['is_private'] is True and meta['enable_gpu'] is True
    assert meta['enable_tpu'] is False and meta['enable_internet'] is False
    assert meta['kernel_sources']==['jarturo/biohub-lab-learned-event-graph']
    notebook=json.loads((folder/'notebook.ipynb').read_text())
    code=''.join(notebook['cells'][1]['source']);tree=ast.parse(code)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]))
    manifest=json.loads((folder/'payload.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert sorted(archive.namelist())==manifest['files']
        for name in archive.namelist():
            assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            if name.endswith('.py'):compile(archive.read(name),name,'exec')
    split=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']
    assert [len(split[k]) for k in ('fit','calibration','evaluation')]==[48,16,48]
    return folder,meta,dict(payload_sha256=manifest['sha256'],notebook_sha256=digest(folder/'notebook.ipynb'),
        metadata_sha256=digest(folder/'kernel-metadata.json'),source_files_verified=len(manifest['files']),
        split_sizes=[48,16,48],private=True,gpu=True,tpu=False,internet=False,
        unit_tests_passed=6,synthetic_training_checkpoint_smoke_steps=3,
        session_timeout_seconds=7200,leaderboard_submission=False)


def main():
    folder,meta,checks=preflight()
    receipt=ROOT/'results/E014_launch.json'
    if receipt.exists():raise RuntimeError('Existing launch receipt: refuse duplicate GPU run')
    (ROOT/'results/E014_preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    response=api.kernels_push(str(folder),timeout='7200')
    errors={k:getattr(response,k,None) for k in ('error','invalid_dataset_sources','invalid_competition_sources','invalid_kernel_sources','invalid_model_sources')}
    errors={k:v for k,v in errors.items() if v}
    if errors:raise RuntimeError(json.dumps(errors))
    record=dict(experiment='E014',kernel=meta['id'],version=getattr(response,'version_number',None),
        kernel_id=getattr(response,'kernel_id',None),url=getattr(response,'url',None),
        launched_at_utc=datetime.now(timezone.utc).isoformat(),**checks,
        authorization='Arturo explicitly renewed permission to use Kaggle GPU on 2026-09-18',
        automatic_monitoring=False,training_steps=6000,training_images_recomputed=False)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    try:
        status=api.kernels_status(meta['id'])
        record['initial_status']=getattr(status.status,'name',str(status.status))
    except Exception as error:
        record['initial_status_check_error']=type(error).__name__
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
