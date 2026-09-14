"""Freeze each independent detector experiment with sources, licenses and hashes."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build(method):
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'src/biohub_official/LICENSE',ROOT/'licenses/CELLECT.txt',
        ROOT/'NOTICE.md',ROOT/'baseline/harmonic_inference.py']
    paths += [ROOT/'scripts'/name for name in ('detector_runner.py','notebook_runner.py','specialist_runner.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.writestr(zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,9,14,0,0,0)),
                p.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    payload=base64.b64encode(buffer.getvalue()).decode()
    sha=hashlib.sha256(buffer.getvalue()).hexdigest()
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
package=Path('/kaggle/working/detector_package')
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
subprocess.run([sys.executable,'-u',str(package/'scripts/detector_runner.py'),{method!r},str(package)],env=env,check=True)
'''
    ast.parse(code)
    title={'cellect':'Biohub Lab CELLECT Detector','gaussian':'Biohub Lab Gaussian Deblending'}[method]
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# {title}\n',
            'Raw-image complementary centers followed by the complete Harmonic linker and official metric.\n',
            'Conditional training diagnostic; not independent validation. No automatic submission.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    meta=json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/'+title.lower().replace(' ','-'),title=title,
        kernel_sources=['jarturo/biohub-lab-official-metric-ab'])
    if method=='cellect':
        meta['dataset_sources'].append('jarturo/biohub-lab-cellect-public-weights')
    folder=ROOT/'kaggle'/('detector_'+method);folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    (folder/'payload.json').write_text(json.dumps(dict(sha256=sha,method=method,
        files=[p.relative_to(ROOT).as_posix() for p in sorted(paths)]),indent=2)+'\n')
    print(method,sha)


if __name__=='__main__':
    build('cellect');build('gaussian')
