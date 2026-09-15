"""E013 companion: full frozen Harmonic on exactly the selector evaluation cohort."""
import hashlib
import json
from pathlib import Path
import time
import traceback
from event_graph_runner import evaluate_dir, run_control
from notebook_runner import competition_dir
from biohub_lab.event_data import save_json


def main(package, root=Path('/kaggle/working/event_control_experiment')):
    package = Path(package); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); path = root/'result.json'
    result = dict(experiment='E013-control', status='verifying_cohort', leaderboard_submitted=False,
                  scope='Full frozen Harmonic on the same 48 E012 development videos as E013 evaluation. Conditional development, not independent validation.')
    save_json(path, result)
    try:
        manifest_path = package/'baseline/event_graph_split.json'
        manifest = json.loads(manifest_path.read_text()); names = manifest['split']['evaluation']
        comp = competition_dir(); inventory = sorted(p.stem for p in (comp/'train').glob('*.zarr'))
        if hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest() != manifest['training_names_sha256']:
            raise ValueError('Training inventory changed')
        if len(names) != 48 or set(names) & {p.stem for p in (comp/'test').glob('*.zarr')}:
            raise ValueError('Invalid control cohort')
        data = evaluate_dir(comp/'train', names, root/'evaluation_data')
        result.update(status='running_full_harmonic', datasets=names,
                      split_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        save_json(path, result); control = run_control(package, data, root); control['scope'] = result['scope']
        result.update(status='complete', control=control, seconds=time.monotonic()-start,
            csv_sha256=hashlib.sha256((root/'harmonic_control/submission.csv').read_bytes()).hexdigest(),
            paired_candidate_kernel='jarturo/biohub-lab-learned-event-graph')
        save_json(path, result); print('EVENT_CONTROL_RESULT', json.dumps(result, indent=2), flush=True)
    except Exception as error:
        result.update(failed_stage=result['status'], status='failed', error=str(error), seconds=time.monotonic()-start)
        save_json(path, result); (root/'error.txt').write_text(traceback.format_exc()); raise


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
