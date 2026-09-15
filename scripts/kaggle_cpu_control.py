"""Repair and evaluate the completed E013 control; no detector or CUDA calls."""
import json
from pathlib import Path
import time
import traceback

from biohub_lab.control_export import E013_CONTROL_CSV_SHA256, repair_export
from biohub_lab.event_data import save_json
from biohub_lab.event_portable import sha
from biohub_lab.evaluate import evaluate_csv, shapes_for
from notebook_runner import competition_dir


def main(package):
    package = Path(package); root = Path('/kaggle/working/control_cpu'); root.mkdir(exist_ok=True)
    started = time.monotonic(); result = dict(status='verifying_inputs', accelerator='none', leaderboard_submitted=False)
    try:
        names = json.loads((package/'baseline/event_graph_split.json').read_text())['split']['evaluation']
        candidates = [Path('/kaggle/input')/prefix/'event_control_experiment/harmonic_control/submission.csv'
            for prefix in ('biohub-lab-event-graph-control', 'notebooks/jarturo/biohub-lab-event-graph-control')]
        candidates += [Path('/kaggle/input')/prefix/'submission.csv' for prefix in (
            'biohub-lab-control-cpu-cache', 'datasets/jarturo/biohub-lab-control-cpu-cache')]
        sources = [p for p in candidates if p.is_file()]
        if len(sources) != 1:
            raise ValueError('Attach the completed E013-control notebook outputs')
        data = root/'evaluation_data'; data.mkdir(exist_ok=True); train = competition_dir()/'train'
        for name in names:
            for suffix in ('.zarr','.geff'):
                target = train/(name+suffix)
                if not target.exists():
                    raise FileNotFoundError(target)
                link = data/target.name
                if not link.exists():
                    link.symlink_to(target, target_is_directory=True)
        corrected = root/'control_cpu_v2.csv'
        result['repair'] = repair_export(sources[0], corrected, shapes_for(data), expected_sha=E013_CONTROL_CSV_SHA256)
        result['status'] = 'evaluating_corrected_control'; save_json(root/'result.json', result)
        metrics = evaluate_csv(corrected, data)
        metrics['scope'] = 'Full Harmonic on 48 E012 development videos; versioned export boundary repair; conditional development'
        save_json(root/'control_metrics.json', metrics)
        result.update(status='complete', control=metrics, csv_sha256=sha(corrected), seconds=time.monotonic()-started)
        save_json(root/'result.json', result); print('CPU_CONTROL_RESULT', json.dumps(result), flush=True)
    except Exception as error:
        result.update(status='failed', error=str(error), seconds=time.monotonic()-started)
        save_json(root/'result.json', result); (root/'error.txt').write_text(traceback.format_exc()); raise


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
