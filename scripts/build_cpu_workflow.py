"""Build new CPU notebooks; never rewrite the frozen E013 v1 notebooks."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(offset=0):
    if not 0 <= offset < 64:
        raise ValueError('Preparation offset must be in the fixed 64-video input cohort')
    names = ['__init__.py','control_export.py','evaluate.py','submission.py','event_data.py','event_portable.py',
        'event_train.py','event_model.py','detector_proposals.py','gaussian_detector.py','patch.py']
    files = [ROOT/'src/biohub_lab'/n for n in names] + list((ROOT/'src/biohub_official').rglob('*.py'))
    files += [ROOT/p for p in ('src/biohub_official/LICENSE','NOTICE.md','baseline/event_graph_split.json',
                              'scripts/kaggle_cpu_control.py','scripts/kaggle_cpu_prepare.py','scripts/notebook_runner.py')]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026,9,15,0,0,0))
            archive.writestr(info,p.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    payload = buffer.getvalue(); digest = hashlib.sha256(payload).hexdigest()
    code = f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['POLARS_PREFER_PKG']='32'
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
package=Path('/kaggle/working/cpu_package');package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()):raise ValueError(name)
    archive.extractall(package)
support=[Path('/kaggle/input/biohub-tracking-support-pack-50ep-v1'),Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1')]
wheels=sorted({{str(p.parent) for root in support if root.is_dir() for p in root.rglob('*.whl')}})
if not wheels:raise RuntimeError('Offline dependency wheels missing')
cmd=[sys.executable,'-m','pip','install','--no-index']
for folder in wheels:cmd+=['--find-links',folder]
subprocess.run(cmd+['tracksdata','polars>=1.36','zarr>=3.0.10,<4','geff-spec<1.2'],check=True)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(package/'src'),str(package/'scripts')]),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1')
subprocess.run([sys.executable,'-u',str(package/'scripts/kaggle_cpu_control.py'),str(package)],env=env,check=True)
'''
    ast.parse(code)
    notebook = dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},cells=[
        dict(id='description',cell_type='markdown',metadata={},source=['# Biohub CPU control recovery\n',
            'Reuse the completed E013-control predictions. Correct one upper-border integer export, preserve all edges and evaluate the official metric on the original 48-video cohort. No GPU, detector rerun, training, or automatic submission.\n']),
        dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    folder=ROOT/'kaggle/cpu_control';folder.mkdir(exist_ok=True)
    meta=dict(id='jarturo/biohub-lab-cpu-control-recovery',title='Biohub Lab CPU Control Recovery',code_file='notebook.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        competition_sources=['biohub-cell-tracking-during-development'],dataset_sources=['pilkwang/biohub-tracking-support-pack-50ep-v1',
            'jarturo/biohub-lab-control-cpu-cache'],kernel_sources=[],model_sources=[])
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (folder/'payload.json').write_text(json.dumps(dict(sha256=digest,files=[p.relative_to(ROOT).as_posix() for p in sorted(files)],
        contains_images=False,contains_weights=False,contains_annotations=False,contains_credentials=False),indent=2)+'\n')
    print(json.dumps(dict(folder=str(folder),payload_sha256=digest,accelerator='none')))
    prepare=json.loads(json.dumps(notebook));prepare_meta=dict(meta)
    prepare_code=code.replace("cmd+['tracksdata','polars>=1.36','zarr>=3.0.10,<4','geff-spec<1.2']", "cmd+['zarr>=3.0.10,<4']")
    prepare_code=prepare_code.replace("scripts/kaggle_cpu_control.py", "scripts/kaggle_cpu_prepare.py")
    if offset:
        prepare_code=prepare_code.replace("str(package)],env=env,check=True)", "str(package),"+repr(str(offset))+"],env=env,check=True)")
    ast.parse(prepare_code)
    prepare['cells'][0]['source']=['# Biohub CPU image preparation\n',
        'Export one verified image shard and prepare Gaussian proposals on CPU. No annotation access. Continue the fixed 48/16/48 experiment in resumable stages on the laptop.\n']
    prepare['cells'][1]['source']=prepare_code.splitlines(True)
    prepare_meta.update(id='jarturo/biohub-lab-cpu-image-preparation',title='Biohub Lab CPU Image Preparation',
        dataset_sources=['pilkwang/biohub-tracking-support-pack-50ep-v1'])
    target=ROOT/'kaggle/cpu_prepare';target.mkdir(exist_ok=True)
    (target/'notebook.ipynb').write_text(json.dumps(prepare,indent=1)+'\n',encoding='utf8')
    (target/'kernel-metadata.json').write_text(json.dumps(prepare_meta,indent=2)+'\n')
    (target/'payload.json').write_bytes((folder/'payload.json').read_bytes())


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--offset',type=int,default=0)
    build(parser.parse_args().offset)
