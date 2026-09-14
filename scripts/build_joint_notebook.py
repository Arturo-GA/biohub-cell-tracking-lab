"""Package the fixed E010 algorithm with exact hashes for cached model outputs."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    paths=[ROOT/'src/biohub_lab'/name for name in
           ('__init__.py','joint_lineage.py','detector_proposals.py','submission.py','evaluate.py','patch.py')]
    paths+=list((ROOT/'src/biohub_official').glob('*.py'))+[ROOT/'src/biohub_official/LICENSE']
    paths+=[ROOT/'NOTICE.md',ROOT/'baseline/harmonic_inference.py',ROOT/'baseline/joint_inputs.json',
            ROOT/'scripts/joint_runner.py',ROOT/'scripts/notebook_runner.py']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,9,14,0,0,0)),
                path.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    payload=base64.b64encode(buffer.getvalue()).decode();sha=hashlib.sha256(buffer.getvalue()).hexdigest()
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
package=Path('/kaggle/working/joint_package')
package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()): raise ValueError(name)
    archive.extractall(package)
env=dict(os.environ,PYTHONPATH=str(package/'src'),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
source=(package/'baseline/harmonic_inference.py').read_text()
marker='\\nmaterialize_inference_repo(ARTIFACTS)'
assert source.count(marker)==1
bootstrap=package/'bootstrap.py'
bootstrap.write_text(source.split(marker)[0])
subprocess.run([sys.executable,'-u',str(bootstrap)],env=env,check=True)
subprocess.run([sys.executable,'-u',str(package/'scripts/joint_runner.py'),str(package)],env=env,check=True)
'''
    ast.parse(code)
    title='Biohub Lab Joint Lineage Selection'
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# {title}\n',
            'E010: joint center and two-daughter path selection over temporal windows, with global event conflicts.\n',
            'Reuses frozen Harmonic graph and completed CELLECT/Gaussian proposal caches; CPU, no new training.\n',
            'Final CSV scored with the pinned official evaluator. Conditional training diagnostic only.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    meta=json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-joint-lineage-selection',title=title,enable_gpu=False,
        kernel_sources=['jarturo/biohub-lab-official-metric-ab','jarturo/biohub-lab-cellect-detector',
                        'jarturo/biohub-lab-gaussian-deblending'])
    meta.pop('machine_shape',None)
    folder=ROOT/'kaggle/joint_lineage';folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    (folder/'payload.json').write_text(json.dumps(dict(sha256=sha,experiment='E010',
        files=[p.relative_to(ROOT).as_posix() for p in sorted(paths)]),indent=2)+'\n')
    print('E010',sha)


if __name__=='__main__':build()
