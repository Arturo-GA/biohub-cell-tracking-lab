"""Package E011 with a frozen, filename-selected development cohort."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    paths=[ROOT/'src/biohub_lab'/name for name in ('__init__.py','detection_dag.py','dag_coverage.py',
        'detector_proposals.py','cellect_detector.py','gaussian_detector.py','temporal_data.py','patch.py')]
    paths+=list((ROOT/'src/biohub_cellect').glob('*.py'))
    paths+=[ROOT/'licenses/CELLECT.txt',ROOT/'NOTICE.md',ROOT/'baseline/harmonic_inference.py',
            ROOT/'baseline/detection_dag_dev.json',ROOT/'scripts/detection_dag_runner.py',ROOT/'scripts/notebook_runner.py']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,9,14,0,0,0)),
                path.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    encoded=base64.b64encode(buffer.getvalue()).decode();sha=hashlib.sha256(buffer.getvalue()).hexdigest()
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
package=Path('/kaggle/working/detection_dag_package');package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()): raise ValueError(name)
    archive.extractall(package)
env=dict(os.environ,PYTHONPATH=str(package/'src'),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
source=(package/'baseline/harmonic_inference.py').read_text()
marker='\\nmaterialize_inference_repo(ARTIFACTS)'
assert source.count(marker)==1
bootstrap=package/'bootstrap.py';bootstrap.write_text(source.split(marker)[0])
subprocess.run([sys.executable,'-u',str(bootstrap)],env=env,check=True)
subprocess.run([sys.executable,'-u',str(package/'scripts/detection_dag_runner.py'),str(package)],env=env,check=True)
'''
    ast.parse(code);title='Biohub Lab Detection DAG Coverage'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# {title}\n',
            'E011: fresh CELLECT/Gaussian detections, factored division pairs, exact disjoint trajectory queries.\n',
            '48 filename-selected development videos, excluding the four prior diagnostics. No Harmonic graph inputs.\n',
            'All proposals and DAGs freeze before annotation access. Coverage is an optimistic ceiling, not accuracy.\n',
            'No new model training, checkpoint selection, official score or submission in this audit.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    meta=json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-detection-dag-coverage',title=title,kernel_sources=[])
    meta['dataset_sources'].append('jarturo/biohub-lab-cellect-public-weights')
    folder=ROOT/'kaggle/detection_dag';folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (folder/'payload.json').write_text(json.dumps(dict(experiment='E011',sha256=sha,
        files=[p.relative_to(ROOT).as_posix() for p in sorted(paths)]),indent=2)+'\n')
    print('E011',sha)


if __name__=='__main__':build()
