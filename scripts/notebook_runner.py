"""Kaggle execution driver, packaged by build_notebooks.py."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from biohub_lab.patch import patch_source


def competition_dir():
    for path in (Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
                 Path('/kaggle/input/biohub-cell-tracking-during-development')):
        if (path / 'test').exists():
            return path
    raise FileNotFoundError('Attach the Biohub competition data')


def diagnostic_data(comp, root, per_prefix=2):
    """Deterministic small TRAIN diagnostic, not claimed as unseen validation."""
    test_ids = {p.stem for p in (comp / 'test').glob('*.zarr')}
    groups = {}
    for p in (comp / 'train').glob('*.zarr'):
        if p.stem not in test_ids:
            groups.setdefault(p.stem.split('_')[0], []).append(p)
    chosen = []
    for prefix, paths in sorted(groups.items()):
        paths.sort(key=lambda p: hashlib.sha256(('biohub-lab-42:' + p.stem).encode()).hexdigest())
        chosen.extend(paths[:per_prefix])
    if not chosen:
        raise RuntimeError('No diagnostic training samples found')
    dest = root / 'diagnostic_data'
    dest.mkdir(exist_ok=True)
    for image in chosen:
        for item in (image, image.with_suffix('.geff')):
            if not item.exists():
                raise FileNotFoundError(item)
            link = dest / item.name
            if not link.exists():
                link.symlink_to(item, target_is_directory=True)
    return dest, [p.stem for p in chosen]


def main(mode, package_root):
    package_root = Path(package_root).resolve()
    root = Path('/kaggle/working')
    comp = competition_dir()
    os.environ['BIOHUB_LAB_SRC'] = str(package_root / 'src')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(package_root / 'src')
    if mode == 'diagnostic':
        data_dir, stems = diagnostic_data(comp, root)
        arms = ['control', 'division_guard']
    else:
        data_dir = comp / 'test'
        stems = sorted(p.stem for p in data_dir.glob('*.zarr'))
        arms = [mode]
    env['BIOHUB_LAB_DATA_DIR'] = str(data_dir)
    source = (package_root / 'baseline/harmonic_inference.py').read_text()
    receipt = {'mode': mode, 'datasets': stems, 'arms': {},
        'scope': 'train diagnostic; checkpoint membership unverified' if mode == 'diagnostic' else 'test inference',
        'quality_status': 'unverified', 'leaderboard_submitted': False}
    for arm in arms:
        arm_root = root / f'biohub_{arm}'
        arm_root.mkdir(exist_ok=True)
        changed = patch_source(source, candidate=(arm == 'division_guard'))
        changed = changed.replace('/kaggle/working', str(arm_root))
        # Score/validate in the same fresh interpreter after dependency setup.
        changed += '\nfrom biohub_lab.evaluate import shapes_for\n'
        changed += 'from biohub_lab.submission import read_and_validate\n'
        changed += 'read_and_validate(SUBMISSION_PATH, shapes_for(TEST_DIR))\n'
        if mode == 'diagnostic':
            changed += 'from biohub_lab.evaluate import evaluate_csv\n'
            changed += 'lab_metrics = evaluate_csv(SUBMISSION_PATH, TEST_DIR)\n'
            changed += 'Path(WORKING_DIR / "official_metrics.json").write_text(json.dumps(lab_metrics, indent=2))\n'
        script = arm_root / 'run.py'
        script.write_text(changed)
        start = time.monotonic()
        subprocess.run([sys.executable, '-u', str(script)], env=env, cwd=arm_root, check=True)
        csv_file = arm_root / 'submission.csv'
        receipt['arms'][arm] = {'seconds': time.monotonic() - start,
            'sha256': hashlib.sha256(csv_file.read_bytes()).hexdigest(), 'csv': str(csv_file)}
        metric_path = arm_root / 'official_metrics.json'
        if metric_path.exists():
            receipt['arms'][arm]['metrics'] = json.loads(metric_path.read_text())
        (root / 'run_receipt.json').write_text(json.dumps(receipt, indent=2))
    if mode != 'diagnostic':
        shutil.copyfile(csv_file, root / 'submission.csv')
    else:
        summaries = {a: receipt['arms'][a]['metrics']['summary'] for a in arms}
        receipt['candidate_minus_control'] = {
            key: summaries['division_guard'][key] - summaries['control'][key]
            for key in ('score', 'adj_edge_jaccard', 'division_jaccard')}
        # A diagnostic never auto-promotes a candidate or emits root submission.csv.
    (root / 'run_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
