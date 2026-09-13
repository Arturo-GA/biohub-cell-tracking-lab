"""Build self-contained offline notebooks from pinned source and our patch."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT / 'baseline/harmonic_inference.py', ROOT / 'baseline/provenance.json',
             ROOT / 'scripts/notebook_runner.py']
    paths.extend(sorted((ROOT / 'src').rglob('*.py')))
    paths.append(ROOT / 'src/biohub_official/LICENSE')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 9, 13, 0, 0, 0))
            z.writestr(info, path.read_bytes().replace(b'\r\n', b'\n'), compress_type=zipfile.ZIP_DEFLATED)
    payload = base64.b64encode(archive.getvalue()).decode()
    checksum = hashlib.sha256(archive.getvalue()).hexdigest()
    metadata = json.loads((ROOT / 'baseline/upstream_metadata.json').read_text())
    for mode, title in [('control', 'Biohub Lab Harmonic Control'),
                        ('division_guard', 'Biohub Lab Division Guard'),
                        ('diagnostic', 'Biohub Lab Official Metric AB')]:
        code = f'''import base64, hashlib, io, os, sys, zipfile, subprocess
from pathlib import Path
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {checksum!r}
package = Path('/kaggle/working/biohub_lab_package')
package.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.namelist():
        target = (package / member).resolve()
        if not target.is_relative_to(package.resolve()):
            raise ValueError('Invalid archive path')
    archive.extractall(package)
env = dict(os.environ, PYTHONPATH=str(package / 'src'))
subprocess.run([sys.executable, '-u', str(package / 'scripts/notebook_runner.py'),
                {mode!r}, str(package)], env=env, check=True)
'''
        ast.parse(code)
        intro = (f'# {title}\n\n'
            'Base: [Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion), '
            'Igor Zharov/flexonafft, with public Pilkwang model artifacts. Source and modifications are pinned in the repository.\n\n'
            'Experiment: reduce reverse association weight near predicted, geometrically plausible divisions. '
            'This is a hypothesis; no improvement or podium position is claimed.\n\n'
            'Official metric snapshot: RoyerLab commit `075fc5f5a52d11077f9dc2b074644618f26939e2`. '
            'The old public proxy-metric sweep is excluded. '
            'Every final CSV is validated for real coordinates and valid lineage topology.\n\n'
            + ('Runs control + candidate on deterministic training samples. **Diagnostic, not proven holdout**: '
               'pretrained checkpoint training membership is not verified. Excludes visible test IDs, evaluates final CSVs, '
               'and does not create a leaderboard submission at the working-directory root.\n'
               if mode == 'diagnostic' else
               'Runs one fixed arm on every test video and writes `/kaggle/working/submission.csv`.\n'))
        nb = {'nbformat': 4, 'nbformat_minor': 5,
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                           'language_info': {'name': 'python', 'version': '3.12'}},
              'cells': [{'id': 'description', 'cell_type': 'markdown', 'metadata': {}, 'source': intro.splitlines(True)},
                        {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                         'outputs': [], 'source': code.splitlines(True)}]}
        folder = ROOT / 'kaggle' / mode
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf8')
        meta = {k: metadata[k] for k in ('language', 'kernel_type', 'dataset_sources', 'competition_sources')}
        meta.update(id='jarturo/' + title.lower().replace(' ', '-'), title=title,
                    code_file='notebook.ipynb', is_private=True, enable_gpu=True,
                    enable_tpu=False, enable_internet=False, kernel_sources=[], model_sources=[],
                    machine_shape='NvidiaTeslaT4')
        (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(mode, folder, 'payload sha256', checksum)


if __name__ == '__main__':
    build()
