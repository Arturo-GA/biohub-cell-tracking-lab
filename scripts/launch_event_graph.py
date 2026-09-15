"""Publish the two authorized private E013 runs, query initial status once, exit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def main():
    preflight = json.loads((ROOT/'results/E013_preflight.json').read_text())
    api = KaggleApi(); api.authenticate()
    for folder_name, experiment in [('event_graph', 'E013'), ('event_graph_control', 'E013-control')]:
        receipt_path = ROOT/'results'/f'{experiment}_launch.json'
        if receipt_path.exists():
            raise RuntimeError('Launch receipt already exists; refuse an accidental duplicate run: '+experiment)
        folder = ROOT/'kaggle'/folder_name
        meta = json.loads((folder/'kernel-metadata.json').read_text())
        payload = json.loads((folder/'payload.json').read_text())
        if not meta['is_private'] or payload['sha256'] != preflight['payload_sha256'][folder_name]:
            raise ValueError('Private metadata or verified payload changed')
        response = api.kernels_push(str(folder))
        errors = {key: getattr(response, key, None) for key in ('error', 'invalid_dataset_sources',
            'invalid_competition_sources', 'invalid_kernel_sources', 'invalid_model_sources')}
        errors = {k: v for k, v in errors.items() if v}
        if errors:
            write(ROOT/'results'/f'{experiment}_launch_error.json', dict(experiment=experiment, errors=errors, at_utc=now()))
            raise RuntimeError(json.dumps(errors))
        record = dict(experiment=experiment, ref=getattr(response, 'ref', None), url=getattr(response, 'url', None),
            versionNumber=getattr(response, 'version_number', None), kernelId=getattr(response, 'kernel_id', None),
            launched_at_utc=now(), payload_sha256=payload['sha256'],
            notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),
            evaluation_videos=48, selector_fit_videos=48 if experiment == 'E013' else 0,
            selector_calibration_videos=16 if experiment == 'E013' else 0,
            local_wait_active=False, leaderboard_submitted=False)
        write(receipt_path, record)
        try:
            status = api.kernels_status(meta['id'])
            record['initial_status'] = getattr(status.status, 'name', str(status.status))
            record['status_checked_at_utc'] = now()
        except Exception as error:
            record['initial_status_error'] = str(error)
        write(receipt_path, record); print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
