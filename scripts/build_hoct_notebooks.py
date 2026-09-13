"""Create standalone offline HOCT diagnostic and complete-test notebooks."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
paths = list((ROOT/'src').rglob('*.py')) + [ROOT/'src/biohub_official/LICENSE',
    ROOT/'scripts/hoct_runner.py',ROOT/'scripts/notebook_runner.py',
    ROOT/'baseline/harmonic_inference.py',ROOT/'licenses/HOCT.txt']
buffer = io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,9,13,0,0,0))
        archive.writestr(info,path.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
payload = base64.b64encode(buffer.getvalue()).decode()
checksum = hashlib.sha256(buffer.getvalue()).hexdigest()
for mode,title in [('diagnostic','Biohub Lab HOCT Morphology Diagnostic'),('test','Biohub Lab HOCT Morphology Submission')]:
    code = f'''import base64, hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {checksum!r}
package = Path('/kaggle/working/hoct_package')
package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.namelist():
        if not (package/member).resolve().is_relative_to(package.resolve()):
            raise ValueError('Unsafe package path')
    archive.extractall(package)
env = dict(os.environ, PYTHONPATH=str(package/'src'))
mode = {mode!r}
if mode == 'diagnostic':
    source = (package/'baseline/harmonic_inference.py').read_text()
    marker = '\\nmaterialize_inference_repo(ARTIFACTS)'
    assert source.count(marker) == 1
    bootstrap = package/'bootstrap.py'
    bootstrap.write_text(source.split(marker)[0])
    subprocess.run([sys.executable,'-u',str(bootstrap)],env=env,check=True)
else:
    subprocess.run([sys.executable,'-u',str(package/'scripts/notebook_runner.py'),
        'control',str(package)],env=env,check=True)
    Path('/kaggle/working/submission.csv').rename('/kaggle/working/control_for_comparison.csv')
subprocess.run([sys.executable,'-u',str(package/'scripts/hoct_runner.py'),
    mode,str(package)],env=env,check=True)
'''
    ast.parse(code)
    intro = f'''# {title}

**Measured result, version 1: negative.** On four in-sample training videos,
this implementation scored 0.9195083 versus 0.9666951 for its fixed-node control.
Evaluated false divisions rose from 1 to 12. Preserved for reproducibility;
**this direct-transfer configuration is not recommended for submission**.

New architecture: image-derived 3D watershed masks → intensity/inertia features →
pretrained [HOCT](https://github.com/royerlab/hoct) general_v1 → exact capacitated matching.
Retains the fixed control detections; replaces its association and division logic.
The masks are approximate intensity basins, not FOCUS-3D predictions.

HOCT commit `2ccc5040823bc944ab67790abd1f56eea7cd4f05`, MIT license included.
Our matcher uses consecutive edges and the fixed-node single-pass objective;
this is **not** the upstream two-pass tracklet solver. FP32 supports Kaggle T4.

{'Four complete training videos, all known to be in the secondary detector training set. This is an in-sample diagnostic, not an unbiased validation score. Reuses cached detections and never exports a root submission.' if mode=='diagnostic' else 'Runs the frozen Harmonic control to obtain all test detections, then replaces all links with HOCT. Exports a validated submission.csv. No improvement is assumed.'}

Detector provenance: public Pilkwang checkpoints and Igor Zharov/flexonafft Harmonic Fusion.
All final CSVs are checked and diagnostic scores use the pinned official metric.
'''
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'id':'description','cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                 {'id':'execute','cell_type':'code','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}]}
    meta = json.loads((ROOT/'kaggle/diagnostic/kernel-metadata.json').read_text())
    meta.update(id='jarturo/'+title.lower().replace(' ','-'),title=title)
    meta['dataset_sources'].append('jarturo/biohub-lab-hoct-public-weights')
    if mode == 'diagnostic':
        meta['kernel_sources'] = ['jarturo/biohub-lab-official-metric-ab']
    folder = ROOT/'kaggle'/f'hoct_{mode}'
    folder.mkdir(exist_ok=True)
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(mode,checksum)
