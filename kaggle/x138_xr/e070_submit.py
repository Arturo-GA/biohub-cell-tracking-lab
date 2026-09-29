"""Submit frozen E070 candidates once, with a live gate for conditional follow-ups."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E070'
COMP = 'biohub-cell-tracking-during-development'
INITIAL = ('C3_fork8', 'F8_dc030')
FALLBACK = ('F8_prune12', 'F8_prune12_joint', 'F8_joint')
ALLOWED = INITIAL + FALLBACK


def load(p): return json.loads(p.read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(p)


def limits(api):
    req = ApiGetSubmissionLimitsRequest()
    req.competition_name = COMP
    with api.build_kaggle_client() as client:
        r = client.competitions.competition_api_client.get_submission_limits(req)
    return dict(num_today=r.num_today, num_allowed_now=r.num_allowed_now,
                limited_by_total=r.limited_by_total)


def require_unfavorable_scores(initial_rows, reference):
    """Missing, failed or still-running scores must never spend follow-up quota."""
    assert len(initial_rows) == 2 and all(r is not None for r in initial_rows), 'Missing initial submission results'
    baseline = Decimal(str(reference))
    assert baseline.is_finite()
    scores = []
    for r in initial_rows:
        state = getattr(r.status, 'name', str(r.status))
        assert state == 'COMPLETE', 'Initial submissions are not both successfully scored: ' + state
        value = Decimal(str(r.public_score))
        assert value.is_finite(), 'Invalid public score'
        scores.append(value)
    assert max(scores) <= baseline, 'A first candidate improved public score; run complementary experiments instead'
    return [str(value) for value in scores]


def submit(api, name):
    assert name in ALLOWED
    is_fallback = name in FALLBACK
    auth_path = RES / ('conditional_next_steps.json' if is_fallback else 'scheduled_first_two.json')
    auth = load(auth_path)
    not_before = auth['not_before_utc'] if is_fallback else auth['scheduled_utc']
    assert datetime.now(timezone.utc) >= datetime.fromisoformat(not_before)
    frozen_rows = auth['authorized_fallback_candidates'] if is_fallback else auth['authorized_candidates']
    frozen = next(r for r in frozen_rows if r['candidate'] == name)
    q = load(RES / 'submission_queue.json')
    row = next(r for r in q['candidates'] if r['candidate'] == name)
    for key in ('kernel', 'version', 'file_name', 'csv_sha256', 'notebook_sha256'):
        assert row[key] == frozen[key], key
    verified = load(Path(row['verification_receipt']))
    assert verified['gate'] == 'PASS' and verified['csv_bounds_and_graph'] == 'PASS'
    assert sha(Path(row['local_file'])) == row['csv_sha256']
    folder = ROOT / 'kaggle/x138_xr' / ('k_c3_fork8' if name == 'C3_fork8' else 'k_e070_' + name.lower())
    assert sha(folder / 'notebook.ipynb') == row['notebook_sha256']
    attempt = RES / f'{name}_submission_attempt.json'
    assert not attempt.exists(), 'Attempt exists: reconcile its ref/status; never automatically resubmit.'
    assert not row['leaderboard_submitted']
    history = [r for r in api.competition_submissions(COMP, page_size=100) or [] if r is not None]
    gate_evidence = None
    if is_fallback:
        by_ref = {str(r.ref): r for r in history}
        references = auth['initial_refs']
        assert references == {'C3_fork8': 56622884, 'F8_dc030': 56622918}
        scores = require_unfavorable_scores([by_ref.get(str(references[n])) for n in INITIAL], auth['reference_public_score'])
        gate_evidence = dict(checked_at_utc=now(), initial_refs=references,
                             public_scores=dict(zip(INITIAL, scores)), reference=auth['reference_public_score'])
        save(RES / 'fallback_gate.json', gate_evidence)
    message = f"E070 reset 20260928 {name} v1 sha={row['csv_sha256'][:16]}"
    assert not any(message in (r.description or '') for r in history), 'Submission already present on Kaggle.'
    assert api.kernels_status(row['kernel']).status.name == 'COMPLETE'
    quota = limits(api)
    assert quota['num_allowed_now'] > 0, quota
    record = dict(candidate=name, kernel=row['kernel'], version=row['version'],
                  csv_sha256=row['csv_sha256'], status='REQUEST_PENDING',
                  attempted_at_utc=now(), message=message, limits_before=quota)
    if is_fallback:
        record['conditional_authorization'] = str(auth_path)
        record['gate_evidence'] = gate_evidence
    with attempt.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    # An uncertain response intentionally leaves REQUEST_PENDING: inspect history before any retry.
    response = api.competition_submit_code(file_name='submission.csv', message=message,
                                          competition=COMP, kernel=row['kernel'],
                                          kernel_version=row['version'], quiet=True)
    record.update(status='ACCEPTED' if response.ref else 'RESPONSE_WITHOUT_REF',
                  ref=response.ref, response_message=response.message, response_at_utc=now())
    save(attempt, record)
    print(json.dumps(record), flush=True)
    assert response.ref, 'No submission ref; inspect durable response, do not blindly retry.'
    row.update(leaderboard_submitted=True, submission_ref=response.ref,
               submission_status='ACCEPTED', submitted_at_utc=record['response_at_utc'])
    q['leaderboard_submissions_made'] = sum(bool(r['leaderboard_submitted']) for r in q['candidates'])
    save(RES / 'submission_queue.json', q)
    ledger = load(RES / 'ledger.json')
    ledger.setdefault('leaderboard_submissions', {})[name] = record
    if name in ledger['runs']:
        ledger['runs'][name].update(leaderboard_submitted=True, submission_ref=response.ref)
    save(RES / 'ledger.json', ledger)
    scope = FALLBACK if is_fallback else INITIAL
    auth['submission_refs'] = [r['submission_ref'] for r in q['candidates'] if r['candidate'] in scope and r.get('submission_ref')]
    auth['status'] = 'SUBMITTED' if len(auth['submission_refs']) == len(scope) else 'PARTIALLY_SUBMITTED'
    save(auth_path, auth)


def status(api, include_fallback=False):
    history = {str(r.ref): r for r in api.competition_submissions(COMP, page_size=100) or [] if r is not None}
    states = []
    for name in (ALLOWED if include_fallback else INITIAL):
        p = RES / f'{name}_submission_attempt.json'
        if not p.exists():
            states.append(dict(candidate=name, status='NOT_ATTEMPTED'))
            continue
        attempt = load(p)
        r = history.get(str(attempt.get('ref')))
        if r is None:
            states.append(dict(candidate=name, status='NEEDS_RECONCILIATION', attempt=attempt))
            continue
        states.append(dict(candidate=name, ref=r.ref, status=getattr(r.status, 'name', str(r.status)),
                           public_score=r.public_score, error_description=r.error_description,
                           date=str(r.date), description=r.description))
    result = dict(checked_at_utc=now(), submissions=states, limits=limits(api))
    save(RES / ('all_submission_status.json' if include_fallback else 'first_two_status.json'), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('submit', 'status', 'status-all'))
    parser.add_argument('candidate', nargs='?', choices=ALLOWED)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.action == 'submit': submit(api, args.candidate)
    else: status(api, include_fallback=args.action == 'status-all')
