"""Build the explicitly authorized private GPU E014 training job."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    names=['src/biohub_lab/__init__.py','src/biohub_lab/association_rank.py',
           'scripts/association_rank_runner.py','baseline/event_graph_split.json','NOTICE.md']
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.writestr(zipfile.ZipInfo(name,date_time=(2026,9,18,0,0,0)),
                (ROOT/name).read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    encoded=base64.b64encode(payload).decode()
    code=f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
package=Path('/kaggle/working/association_rank_package');package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()):raise ValueError(name)
    archive.extractall(package)
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(package/'src'),str(package/'scripts')]),
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
command='from pathlib import Path;from association_rank_runner import main;main(Path('+repr(str(package))+'))'
subprocess.run([sys.executable,'-u','-c',command],env=env,check=True)
'''
    ast.parse(code)
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        cells=[dict(id='description',cell_type='markdown',metadata={},source=[
            '# Biohub E014 Association Ranking\n',
            'Private GPU training authorized by Arturo on 2026-09-18.\n',
            'Reuses pinned E013 appearance and detection graphs: no image inference.\n',
            '48 fit videos, 16 calibration videos, 6000 steps; separate link/division encoders.\n',
            'Balanced known-label classification plus within-mother ranking. Unknown labels remain masked.\n',
            'No evaluation-label access or leaderboard submission. Ranking diagnostics are not competition scores.\n']),
            dict(id='run',cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)])
    metadata=dict(id='jarturo/biohub-lab-association-ranking',title='Biohub Lab Association Ranking',
        code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
        dataset_sources=[],competition_sources=['biohub-cell-tracking-during-development'],
        kernel_sources=['jarturo/biohub-lab-learned-event-graph'],model_sources=[])
    folder=ROOT/'kaggle/association_rank';folder.mkdir(exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),
                       ('payload.json',dict(experiment='E014',sha256=digest,files=sorted(names)))]:
        (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(digest)


if __name__=='__main__':build()
