"""Execute and verify final private notebooks; never submit to the leaderboard."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
import threading
import time
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E071'
ACTIVE = {'QUEUED', 'RUNNING', 'CANCEL_PENDING'}


def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(p)


def ledger():
    p = RES / 'ledger.json'
    return load(p) if p.exists() else dict(experiment='E071', runs={}, auto_submit=False,
        reminder_enabled=False, target_ready=5, base_public_score=0.955)


def occupancy(api):
    rows = []
    # Recent active runs in other projects must also be counted, including queued ones.
    for r in api.kernels_list(mine=True, sort_by='dateRun', page_size=30) or []:
        if not r or not r.ref: continue
        try:
            s = api.kernels_status(r.ref).status.name
        except Exception as exc:
            if getattr(getattr(exc, 'response', None), 'status_code', None) == 404:
                continue  # Published notebook without a corresponding session.
            raise
        if s in ACTIVE:
            rows.append(dict(kernel=r.ref, status=s))
    save(RES / 'active_sessions.json', dict(checked_at_utc=now(), active=rows))
    return rows


def launch(api, name):
    plans = {r['candidate']: r for r in load(RES / 'candidate_plan.json')['candidates']}
    row = plans[name]
    current = ledger()
    if current.get('execution_mode') == 'LOCAL_GPU_PREFERRED':
        assert not current.get('remaining_kaggle_launches_on_hold'), 'Final registration is on hold; finish local validation first.'
        local_receipt = ROOT / 'outputs/e071/local' / name / 'verified.json'
        assert local_receipt.exists(), 'Do not spend GPU on an unverified local candidate'
        assert load(local_receipt)['stage'] == 'LOCAL_VERIFIED_ONLY'
        quota = api.quota_view().to_dict(); save(RES / 'accelerator_quota_latest.json', quota)
        gpu = quota['gpuQuota']
        # The SDK currently serializes fractional durations as e.g. 11501.1.0s.
        def seconds(v):
            assert str(v).endswith('s'), 'Unexpected quota duration unit'
            return float(re.match(r'^\d+(?:\.\d+)?', str(v)).group(0))
        remaining = seconds(gpu['totalTimeAllowed']) - seconds(gpu['timeUsed']) - seconds(gpu.get('timeReserved','0s'))
        assert remaining >= 4800, 'Keep at least one hour after the estimated final validation run'
    path = RES / f'{name}_launch.json'
    assert not path.exists(), 'Already attempted. Reconcile receipt instead of launching again.'
    active = occupancy(api)
    assert len(active) < 2, ('Shared two-session ceiling', active)
    folder = Path(row['folder'])
    meta = load(folder / 'kernel-metadata.json')
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert sha(folder / meta['code_file']) == row['notebook_sha256']
    record = dict(candidate=name, kernel=meta['id'], folder=str(folder), gpu=True,
                  notebook_sha256=row['notebook_sha256'], time_utc=now(),
                  status='REQUEST_PENDING', version=None, leaderboard_submitted=False,
                  other_sessions_at_launch=active)
    # Durable before the mutating request: uncertain responses must not be repeated.
    with path.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    state = ledger(); state['runs'][name] = record; save(RES / 'ledger.json', state)
    response = api.kernels_push(str(folder)).to_dict()
    record.update(response=response, version=response.get('versionNumber'), response_at_utc=now(),
                  status='LAUNCHED' if response.get('versionNumber') and not response.get('error') else 'LAUNCH_FAILED')
    save(path, record)
    state['runs'][name] = record; save(RES / 'ledger.json', state)
    print(json.dumps(record), flush=True)
    assert record['status'] == 'LAUNCHED', response


def check(api):
    state = ledger(); changes = []
    for name, row in state['runs'].items():
        if not row.get('version'): continue
        status = api.kernels_status(row['kernel']).status.name
        if row['status'] != status:
            changes.append(dict(candidate=name, old=row['status'], new=status))
        row['status'] = status
    state['checked_at_utc'] = now(); save(RES / 'ledger.json', state)
    print(json.dumps(dict(checked_at_utc=state['checked_at_utc'], changes=changes,
                         states={n:r['status'] for n,r in state['runs'].items()})), flush=True)
    return state


def download(api, name):
    record = load(RES / f'{name}_launch.json')
    assert api.kernels_status(record['kernel']).status.name == 'COMPLETE'
    folder = Path(record['folder'])
    assert sha(folder / 'notebook.ipynb') == record['notebook_sha256']
    out = ROOT / 'outputs/e071' / name
    api.kernels_output(record['kernel'], str(out),
        file_pattern=r'(submission\.csv$|run_manifest\.json$|\.log$)', quiet=True)
    logs = list(out.glob('*.log')); assert len(logs) == 1
    raw = logs[0].read_text(encoding='utf-8')
    events = json.loads(raw)
    assert not any('readmit skipped' in e.get('data','') for e in events), 'Readmission silently failed'
    subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'kaggle/x138_xr/verify_run.py'),
        str(folder), str(out), str(RES / f'{name}_launch.json'), str(logs[0]),
        '--save', str(RES / f'{name}_verified.json')], check=True)
    verified = load(RES / f'{name}_verified.json')
    verified['duration_seconds'] = max(float(e.get('time',0)) for e in events)
    local_receipt = ROOT / 'outputs/e071/local' / name / 'verified.json'
    if local_receipt.exists():
        # The platform runtime does not include numpy; compare in the local ML environment.
        comparison = subprocess.run([str(ROOT / '.venv-e071/Scripts/python.exe'),
            str(ROOT / 'kaggle/x138_xr/e071_compare_remote.py'), name],check=True,capture_output=True,text=True)
        verified['local_replay_parity'] = json.loads(comparison.stdout)
    save(RES / f'{name}_verified.json', verified)
    state = ledger(); state['runs'][name].update(status='COMPLETE', download_verified=True,
        csv_sha256=verified['csv_sha256']); save(RES / 'ledger.json', state)
    print(json.dumps(verified), flush=True)


def logs(api, name, seconds):
    row = ledger()['runs'][name]
    path = ROOT / 'outputs/e071/streams' / (name + '.txt'); path.parent.mkdir(parents=True, exist_ok=True)
    def stream():
        try:
            with path.open('w', encoding='utf-8') as f:
                for event in api.kernels_logs_stream(row['kernel']):
                    chunk = event.get('data',''); f.write(chunk); f.flush()
        except Exception as exc:
            print(type(exc).__name__, str(exc)[:300], flush=True)
    t = threading.Thread(target=stream, daemon=True); t.start(); t.join(seconds)
    if path.exists():
        lines = path.read_text(encoding='utf-8').splitlines()
        relevant = [l for l in lines if any(k in l for k in ['E071','Traceback','REPAIR FAILED','readmitted','Final submission','Configuration guard','CUDA','RuntimeError'])]
        print('\n'.join(relevant[-18:]), flush=True)


def run_batch(api):
    """One finite preparation job, not a scheduled reminder or leaderboard sender."""
    lock = RES / 'batch.lock'
    with lock.open('x', encoding='utf-8') as f: f.write(now())
    deadline = time.monotonic() + 3 * 3600
    try:
        from finalize_e071 import compare
        plans = load(RES / 'candidate_plan.json')['candidates']
        while time.monotonic() < deadline:
            state = check(api)
            for name, row in state['runs'].items():
                if row['status'] == 'COMPLETE' and not row.get('download_verified'):
                    download(api, name)
                    compare()
                elif row['status'] in {'ERROR','CANCEL_ACKNOWLEDGED','CANCELED','LAUNCH_FAILED','REQUEST_PENDING'}:
                    raise RuntimeError(f'{name}: {row["status"]}; inspect receipt/logs before continuing')
            state = ledger()
            if all(state['runs'].get(p['candidate'],{}).get('download_verified') for p in plans):
                assert (RES / 'submission_queue.json').exists(), 'Distinct-output queue not frozen'
                print('E071 FIVE_READY; finite preparation job completed; no leaderboard sends.', flush=True)
                return
            pending = [p['candidate'] for p in plans if p['candidate'] not in state['runs']]
            if pending:
                active = occupancy(api)
                if len(active) < 2:
                    launch(api, pending[0])
                    continue
            # Sleep in bounded increments; the process only handles this finite batch.
            time.sleep(30)
            time.sleep(30)
        raise TimeoutError('Three-hour preparation limit reached; inspect ledger before continuing')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['launch','check','download','occupancy','logs','run-batch'])
    p.add_argument('name', nargs='?'); p.add_argument('--seconds', type=int, default=15)
    args = p.parse_args(); api = KaggleApi(); api.authenticate()
    if args.action == 'launch': launch(api,args.name)
    elif args.action == 'download': download(api,args.name)
    elif args.action == 'check': check(api)
    elif args.action == 'logs': logs(api,args.name,args.seconds)
    elif args.action == 'run-batch': run_batch(api)
    else: print(json.dumps(occupancy(api)), flush=True)
