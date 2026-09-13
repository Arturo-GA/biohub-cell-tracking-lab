"""Freeze reviewed public inputs, keeping origin and SHA256 receipts."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    src = ROOT / 'reference/harmonic/biohub-harmonic-fusion.ipynb'
    nb = json.loads(src.read_text(encoding='utf8'))
    # Only the inference stage. Later public cells implement a proxy scorer
    # and a training-data sweep, neither used by our fixed control/candidate.
    chunks = []
    for cell in nb['cells'][:7]:
        text = ''.join(cell['source'])
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        for item in reversed(tree.body):
            if isinstance(item, ast.ImportFrom) and item.module == '__future__':
                del lines[item.lineno - 1:item.end_lineno]
        chunks.append(''.join(lines))
    source = 'from __future__ import annotations\nfrom IPython.display import display\n' + '\n\n'.join(chunks)
    source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
    compile(source, 'harmonic_inference.py', 'exec')
    baseline = ROOT / 'baseline'
    baseline.mkdir(exist_ok=True)
    (baseline / 'harmonic_inference.py').write_text(source, encoding='utf8')
    shutil.copyfile(ROOT / 'reference/harmonic/kernel-metadata.json', baseline / 'upstream_metadata.json')
    dest = ROOT / 'src/biohub_official'
    dest.mkdir(exist_ok=True)
    (dest / '__init__.py').write_text('"""Pinned RoyerLab metric; see LICENSE."""\n')
    for name in ('metrics.py', 'division_metrics.py'):
        original = ROOT / 'third_party/official/src/tracking_cellmot' / name
        (dest / name).write_bytes(original.read_text(encoding='utf8').encode('utf8'))
    (dest / 'LICENSE').write_bytes((ROOT / 'third_party/official/LICENSE').read_text(encoding='utf8').encode('utf8'))
    receipt = {
        'retrieved': '2026-09-13',
        'harmonic_url': 'https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion',
        'harmonic_notebook_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
        'inference_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'included_cells': list(range(7)),
        'excluded_cells_reason': 'Old proxy metric and unverified holdout selection/sweep',
        'user_notebook_url': 'https://www.kaggle.com/code/amanatar/improved-metric-hack-last-call',
        'official_repository': 'https://github.com/royerlab/kaggle-cell-tracking-competition',
        'official_commit': '075fc5f5a52d11077f9dc2b074644618f26939e2',
        'official_license': 'BSD-3-Clause',
        'metric_sha256': hashlib.sha256((dest / 'metrics.py').read_bytes()).hexdigest(),
        'division_metric_sha256': hashlib.sha256((dest / 'division_metrics.py').read_bytes()).hexdigest(),
    }
    (baseline / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
