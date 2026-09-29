"""Send only Arturo's five frozen final E071 candidates, once each."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from e070_submit import limits, sha, save, load, now, COMP

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E071'
ALLOWED = ('C3_D04_R094', 'C3_R094', 'C3_D04', 'C3_D08_R094', 'C3_D04_R092')
IDENTITY = ('candidate','kernel','version','file_name','csv_sha256','notebook_sha256')


def history(api):
    return [r for r in api.competition_submissions(COMP,page_size=100) or [] if r is not None]


def description(row):
    return f"E071 final 20260929 {row['candidate']} v{row['version']} sha={row['csv_sha256'][:16]}"


def candidate_in_history(records,name):
    # C3_D04 is also a prefix of C3_D04_R094: require a whole identifier.
    pattern=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])')
    return any('E071' in (r.description or '') and pattern.search(r.description or '') for r in records)


def validate_local(row):
    auth=load(RES/'final_five_authorization.json')
    assert datetime.now(timezone.utc)>=datetime.fromisoformat(auth['not_before_utc'])
    frozen=next(r for r in auth['candidates'] if r['candidate']==row['candidate'])
    assert row['candidate'] in ALLOWED
    for key in IDENTITY:assert frozen[key]==row[key],key
    verified=load(Path(row['verification_receipt']))
    assert verified['gate']==verified['csv_bounds_and_graph']=='PASS'
    assert len(verified['local_replay_parity'])==4 and all(verified['local_replay_parity'].values())
    for key in ('candidate','kernel','version','csv_sha256','notebook_sha256'):
        assert verified[key]==row[key],key
    assert sha(Path(row['local_file']))==row['csv_sha256']
    assert sha(Path(row['folder'])/'notebook.ipynb')==row['notebook_sha256']
    assert row['file_name']=='submission.csv' and row['version']==1


def preflight(api):
    queue=load(RES/'submission_queue.json');rows=queue['candidates']
    assert [r['candidate'] for r in rows]==list(ALLOWED)
    existing=history(api);quota=limits(api);checks=[]
    assert quota['num_allowed_now']>=5,quota
    for row in rows:
        validate_local(row)
        name=row['candidate']
        assert not row['leaderboard_submitted']
        assert not (RES/f'{name}_submission_attempt.json').exists(),'Reconcile an existing attempt before continuing'
        assert not candidate_in_history(existing,name)
        state=api.kernels_status(row['kernel']).status.name
        assert state=='COMPLETE',(name,state)
        folder=ROOT/'outputs/e071/submission_remote_code'/name
        folder.mkdir(parents=True,exist_ok=True)
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=row['kernel'].split('/')
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel(req)
        meta=response.metadata.to_dict()
        assert meta['currentVersionNumber']==row['version'] and meta['ref']==row['kernel']
        save(folder/'remote-metadata.json',meta)
        (folder/'notebook.ipynb').write_text(response.blob.source,encoding='utf8')
        remote=json.loads(response.blob.source);local=load(Path(row['folder'])/'notebook.ipynb')
        cells=lambda nb:[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        assert cells(remote)==cells(local),(name,'remote v1 source mismatch')
        assert meta['isPrivate'] and meta['enableGpu'] and not meta['enableInternet']
        checks.append(dict(candidate=name,kernel=row['kernel'],version=row['version'],
                           remote_code_equal=True,local_hashes_valid=True,status=state))
    result=dict(checked_at_utc=now(),limits=quota,checks=checks,passed=True)
    save(RES/'final_preflight.json',result);print(json.dumps(result),flush=True)


def submit(api,name):
    assert name in ALLOWED
    pre=load(RES/'final_preflight.json');assert pre['passed'] and len(pre['checks'])==5
    queue=load(RES/'submission_queue.json');row=next(r for r in queue['candidates'] if r['candidate']==name)
    validate_local(row)
    attempt_path=RES/f'{name}_submission_attempt.json'
    assert not attempt_path.exists(),'Existing attempt: reconcile; never submit blindly again'
    assert not row['leaderboard_submitted']
    existing=history(api)
    assert not candidate_in_history(existing,name),'Candidate already in Kaggle history'
    assert api.kernels_status(row['kernel']).status.name=='COMPLETE'
    quota=limits(api);assert quota['num_allowed_now']>0,quota
    message=description(row)
    attempt=dict(candidate=name,kernel=row['kernel'],version=row['version'],file_name=row['file_name'],
        csv_sha256=row['csv_sha256'],notebook_sha256=row['notebook_sha256'],
        status='REQUEST_PENDING',attempted_at_utc=now(),limits_before=quota,message=message,
        authorization='results/E071/final_five_authorization.json')
    with attempt_path.open('x',encoding='utf8') as f:json.dump(attempt,f,indent=2)
    # A timeout or ambiguous response leaves this receipt intact. Do not repeat the call.
    response=api.competition_submit_code(file_name=row['file_name'],message=message,
        competition=COMP,kernel=row['kernel'],kernel_version=row['version'],quiet=True)
    attempt.update(status='ACCEPTED' if response.ref else 'RESPONSE_WITHOUT_REF',ref=response.ref,
        response_message=response.message,response_at_utc=now())
    save(attempt_path,attempt);print(json.dumps(attempt),flush=True)
    assert response.ref,'No submission identifier: reconcile response and history before any action'
    row.update(leaderboard_submitted=True,submission_ref=response.ref,
        submission_status='ACCEPTED',submitted_at_utc=attempt['response_at_utc'])
    queue['leaderboard_submissions_made']=sum(bool(r['leaderboard_submitted']) for r in queue['candidates'])
    queue['status']='FIVE_SUBMITTED' if queue['leaderboard_submissions_made']==5 else 'PARTIALLY_SUBMITTED'
    save(RES/'submission_queue.json',queue)
    ledger=load(RES/'ledger.json');ledger.setdefault('leaderboard_submissions',{})[name]=attempt
    ledger['runs'][name].update(leaderboard_submitted=True,submission_ref=response.ref)
    ledger['leaderboard_submissions_made']=queue['leaderboard_submissions_made'];ledger['status']=queue['status']
    save(RES/'ledger.json',ledger)
    auth=load(RES/'final_five_authorization.json');auth['status']=queue['status']
    auth['submission_refs']={r['candidate']:r['submission_ref'] for r in queue['candidates'] if r.get('submission_ref')}
    save(RES/'final_five_authorization.json',auth)


def status(api):
    records=history(api);by_ref={str(r.ref):r for r in records}
    rows=[];queue=load(RES/'submission_queue.json');ledger=load(RES/'ledger.json')
    for row in queue['candidates']:
        name=row['candidate'];p=RES/f'{name}_submission_attempt.json'
        if not p.exists():rows.append(dict(candidate=name,status='NOT_ATTEMPTED'));continue
        attempt=load(p);remote=by_ref.get(str(attempt.get('ref')))
        if remote is None:
            matches=[r for r in records if (r.description or '')==attempt['message']]
            rows.append(dict(candidate=name,status='NEEDS_RECONCILIATION',matching_refs=[r.ref for r in matches]));continue
        state=getattr(remote.status,'name',str(remote.status))
        info=dict(candidate=name,ref=remote.ref,status=state,public_score=remote.public_score,
            error_description=remote.error_description,date=str(remote.date),description=remote.description)
        rows.append(info)
        row.update(submission_status=state,public_score=remote.public_score)
        ledger['runs'][name].update(submission_status=state,public_score=remote.public_score)
    report=dict(checked_at_utc=now(),limits=limits(api),submissions=rows)
    save(RES/'all_submission_status.json',report)
    save(RES/'submission_queue.json',queue);save(RES/'ledger.json',ledger)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','submit','status-all'])
    p.add_argument('candidate',nargs='?',choices=ALLOWED);args=p.parse_args()
    api=KaggleApi();api.authenticate()
    if args.action=='preflight':preflight(api)
    elif args.action=='submit':submit(api,args.candidate)
    else:status(api)
