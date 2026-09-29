"""Verify a downloaded inference run against its frozen notebook and launch receipt.

This never uploads or submits. Use only with output from a COMPLETE Kaggle run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from biohub_lab.submission import read_and_validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder, output, receipt, log):
    launch = json.loads(receipt.read_text(encoding='utf-8'))
    meta = json.loads((folder / 'kernel-metadata.json').read_text(encoding='utf-8'))
    notebook = folder / meta['code_file']
    assert launch['kernel'] == meta['id'] and launch['version'] is not None
    assert launch['notebook_sha256'] == sha(notebook), 'Notebook changed since launch'
    expected = json.loads((folder / 'expected_manifest.json').read_text(encoding='utf-8'))
    actual = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
    assert actual == expected, 'Downloaded run does not match candidate manifest'
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    # build_c3.py appends the manifest as the last cell, after hashing source.
    code = '\n'.join(''.join(c['source']) for c in nb['cells'][:-1] if c['cell_type'] == 'code')
    assert hashlib.sha256(code.encode()).hexdigest() == actual['source_sha256']
    shapes = json.loads((ROOT / 'results/E054_CONTROL_completed.json').read_text(encoding='utf-8'))['result']['shapes']
    groups = read_and_validate(output / 'submission.csv', shapes)
    targets = ','.join(f'{s}={len(n)}/{len(e)}' for s, (n, e) in sorted(groups.items()))
    subprocess.run([sys.executable, str(Path(__file__).with_name('gate.py')), str(log), targets, '0'], check=True)
    return dict(kernel=launch['kernel'], version=launch['version'], candidate=actual['candidate'],
                source_sha256=actual['source_sha256'], notebook_sha256=sha(notebook),
                csv_sha256=sha(output / 'submission.csv'), rows=sum(len(n)+len(e) for n,e in groups.values()),
                counts=targets.split(','), gate='PASS', csv_bounds_and_graph='PASS',
                leaderboard_submitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('folder', 'output', 'receipt', 'log'):
        p.add_argument(name, type=Path)
    p.add_argument('--save', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.folder, a.output, a.receipt, a.log)
    a.save.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
