"""Package the temporal experiment and its offline bootstrap for Kaggle."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build(mode):
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'src/biohub_official/LICENSE',
        ROOT/'baseline/harmonic_inference.py',ROOT/'scripts/audit_temporal_data.py']
    for name in ('temporal_runner.py','notebook_runner.py'):
        if (ROOT/'scripts'/name).exists(): paths.append(ROOT/'scripts'/name)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.writestr(zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,9,14,0,0,0)),
                p.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    payload=base64.b64encode(buf.getvalue()).decode()
    sha=hashlib.sha256(buf.getvalue()).hexdigest()
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
package=Path('/kaggle/working/temporal_package')
package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()): raise ValueError(name)
    archive.extractall(package)
env=dict(os.environ,PYTHONPATH=str(package/'src'))
source=(package/'baseline/harmonic_inference.py').read_text()
marker='\\nmaterialize_inference_repo(ARTIFACTS)'
assert source.count(marker)==1
bootstrap=package/'bootstrap.py'
bootstrap.write_text(source.split(marker)[0])
subprocess.run([sys.executable,'-u',str(bootstrap)],env=env,check=True)
script={('audit_temporal_data.py' if mode=='audit' else 'temporal_runner.py')!r}
subprocess.run([sys.executable,'-u',str(package/'scripts'/script),{mode!r},str(package)],env=env,check=True)
'''
    ast.parse(code)
    title='Biohub Lab Temporal '+mode.title()
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# {title}\n',
            'Own temporal association experiment. Split provenance and evaluation scope are recorded explicitly.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    meta=json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/'+title.lower().replace(' ','-').replace('_','-'),title=title,enable_gpu=mode not in ('audit','prepare','train_cpu'))
    if mode in ('audit','prepare','train_cpu'): meta.pop('machine_shape',None)
    if mode in ('train','train_cpu'): meta['kernel_sources']=['jarturo/biohub-lab-temporal-prepare','jarturo/biohub-lab-official-metric-ab']
    folder=ROOT/'kaggle'/('temporal_'+mode)
    folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    print(mode,sha,flush=True)


if __name__=='__main__':
    import sys
    build(sys.argv[1])
