"""Deterministic E013 private notebook payload; no annotations or weights embedded."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT/'src/biohub_lab'/name for name in (
        '__init__.py', 'event_data.py', 'event_model.py', 'event_train.py', 'event_inference.py', 'event_solver.py',
        'harmonic_centers.py', 'detection_dag.py', 'detector_proposals.py', 'temporal_data.py',
        'cellect_detector.py', 'gaussian_detector.py', 'patch.py', 'evaluate.py', 'submission.py')]
    paths += list((ROOT/'src/biohub_cellect').rglob('*.py'))
    paths += list((ROOT/'src/biohub_official').rglob('*.py'))
    paths += [ROOT/'src/biohub_official/LICENSE']
    paths += [ROOT/p for p in ('NOTICE.md', 'licenses/CELLECT.txt', 'baseline/harmonic_inference.py',
        'baseline/event_graph_split.json', 'baseline/event_graph_inputs.json', 'scripts/event_graph_runner.py',
        'scripts/notebook_runner.py', 'scripts/event_control_runner.py')]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 9, 15, 0, 0, 0)),
                             path.read_bytes().replace(b'\r\n', b'\n'), compress_type=zipfile.ZIP_DEFLATED)
    encoded = base64.b64encode(buffer.getvalue()).decode(); digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    suffix = '\nfrom event_graph_runner import main\nmain(PACKAGE, REPO_DIR, _primary_materialized_path, SECONDARY_WEIGHTS_PATH)\n'
    code = f'''import base64,hashlib,io,os,subprocess,sys,zipfile
from pathlib import Path
payload=base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
package=Path('/kaggle/working/event_graph_package');package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (package/name).resolve().is_relative_to(package.resolve()):raise ValueError(name)
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
    ast.parse(code)
    title = 'Biohub Lab Learned Event Graph'
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        cells=[dict(id='description', cell_type='markdown', metadata={}, source=[f'# {title}\n',
            'E013: 48 fresh selector-fit videos, 16 calibration videos, 48 E012 development evaluation videos.\n',
            'Frozen dual-UNet visual features + local graph attention, explicit divisions and overlapping temporal MILP.\n',
            'Sparse labels on real detections; all evaluation predictions freeze before evaluation annotation access.\n',
            'Companion E013-control runs full Harmonic on the same 48 videos. Conditional development, not independent validation.\n',
            'Long run: outputs and checkpoints are saved per stage. No automatic leaderboard submission.\n']),
            dict(id='run', cell_type='code', metadata={}, source=code.splitlines(True), outputs=[], execution_count=None)])
    meta = json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/biohub-lab-learned-event-graph', title=title,
                kernel_sources=['jarturo/biohub-lab-harmonic-detection-dag'])
    meta['dataset_sources'].append('jarturo/biohub-lab-cellect-public-weights')
    folder = ROOT/'kaggle/event_graph'; folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    (folder/'payload.json').write_text(json.dumps(dict(experiment='E013', sha256=digest,
        files=[p.relative_to(ROOT).as_posix() for p in sorted(paths)]), indent=2)+'\n')
    control = json.loads(json.dumps(notebook))
    control['cells'][0]['source'] = ['# Biohub Lab Event Graph Control\n',
        'E013-control: unchanged full Harmonic on exactly the 48 E013 evaluation videos.\n',
        'Run separately to keep each experiment within a Kaggle session. No model fitting or leaderboard submission.\n']
    boundary = "source=(package/'baseline/harmonic_inference.py').read_text()"
    control_code = code.split(boundary)[0]
    control_code += "subprocess.run([sys.executable,'-u',str(package/'scripts/event_control_runner.py'),str(package)],env=env,check=True)\n"
    ast.parse(control_code)
    control['cells'][1]['source'] = control_code.splitlines(True)
    control_meta = dict(meta, id='jarturo/biohub-lab-event-graph-control', title='Biohub Lab Event Graph Control', kernel_sources=[])
    control_meta['dataset_sources'] = [s for s in meta['dataset_sources'] if not s.startswith('jarturo/')]
    folder_control = ROOT/'kaggle/event_graph_control'; folder_control.mkdir(exist_ok=True)
    (folder_control/'notebook.ipynb').write_text(json.dumps(control, indent=1), encoding='utf8')
    (folder_control/'kernel-metadata.json').write_text(json.dumps(control_meta, indent=2)+'\n')
    (folder_control/'payload.json').write_bytes((folder/'payload.json').read_bytes())
    print('E013', digest)


if __name__ == '__main__':
    build()
