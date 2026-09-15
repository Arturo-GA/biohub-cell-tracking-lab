"""Package E012 with unchanged E011 settings and hashes of reusable proposals."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def bootstrap_source(source):
    anchor='print("secondary edge-feature TTA patch installed and enabled", flush=True)'
    if source.count(anchor)!=1:raise ValueError('Harmonic initialization boundary changed')
    return source.split(anchor)[0]+anchor+'\n'


def build():
    paths=[ROOT/'src/biohub_lab'/name for name in ('__init__.py','detection_dag.py','dag_coverage.py',
        'detector_proposals.py','harmonic_centers.py','temporal_data.py','patch.py')]
    paths+=[ROOT/'NOTICE.md',ROOT/'baseline/harmonic_inference.py',ROOT/'baseline/detection_dag_dev.json',
        ROOT/'baseline/harmonic_dag_inputs.json',ROOT/'scripts/harmonic_dag_runner.py',ROOT/'scripts/notebook_runner.py']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,9,15,0,0,0)),
                path.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    encoded=base64.b64encode(buffer.getvalue()).decode();digest=hashlib.sha256(buffer.getvalue()).hexdigest()
    prefix=bootstrap_source((ROOT/'baseline/harmonic_inference.py').read_text())
    suffix='''
from harmonic_dag_runner import main as run_harmonic_dag
run_harmonic_dag(PACKAGE, REPO_DIR, _primary_materialized_path, SECONDARY_WEIGHTS_PATH)
'''
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
package=Path('/kaggle/working/harmonic_dag_package');package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()): raise ValueError(name)
    archive.extractall(package)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(package/'src'),str(package/'scripts')]),
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
source=(package/'baseline/harmonic_inference.py').read_text()
anchor='print("secondary edge-feature TTA patch installed and enabled", flush=True)'
assert source.count(anchor)==1
bootstrap=package/'bootstrap.py'
bootstrap.write_text(source.split(anchor)[0]+anchor+'\\nPACKAGE=Path('+repr(str(package))+')\\n'+{suffix!r})
subprocess.run([sys.executable,'-u',str(bootstrap)],env=env,check=True)
'''
    ast.parse(prefix+'\nPACKAGE=Path("/tmp/package")\n'+suffix);ast.parse(code)
    title='Biohub Lab Harmonic Detection DAG'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# {title}\n',
            'E012: primary dual-seed image detections, with associations removed, plus frozen E011 proposals.\n',
            'Paired E011 / Harmonic-only / combined coverage on the same 48 development videos and fixed criteria.\n',
            'All new graphs freeze before annotation access. No training, official score or submission.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    meta=json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-harmonic-detection-dag',title=title,
        kernel_sources=['jarturo/biohub-lab-detection-dag-coverage'])
    folder=ROOT/'kaggle/harmonic_dag';folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (folder/'payload.json').write_text(json.dumps(dict(experiment='E012',sha256=digest,
        files=[p.relative_to(ROOT).as_posix() for p in sorted(paths)]),indent=2)+'\n')
    print('E012',digest)


if __name__=='__main__':build()
