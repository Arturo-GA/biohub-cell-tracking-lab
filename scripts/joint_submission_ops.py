"""Explicit E054 validation, one durable submission attempt, or score read."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
COMP='biohub-cell-tracking-during-development'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
def main(action,stage='E054_SUBMISSION'):
    assert stage in ('E054_SUBMISSION','E054_CONTROL')
    api=KaggleApi();api.authenticate();prefix=ROOT/('results/'+stage)
    launch_path=Path(str(prefix)+'_launch.json');launch=read(launch_path)
    done_path=Path(str(prefix)+'_completed.json');attempt_path=Path(str(prefix)+'_attempt.json')
    if action=='check':
        state=api.kernels_status(launch['kernel']).status.name;print(state,flush=True)
        if state!='COMPLETE':return
        out=ROOT/'outputs'/stage.lower();api.kernels_output(launch['kernel'],str(out),file_pattern=r'(^submission.csv$|^joint_submission/result.json$)',quiet=True)
        r=read(out/'joint_submission/result.json');assert r['status']=='complete' and sha(out/'submission.csv')==r['csv_sha256']
        from biohub_lab.submission import read_and_validate
        groups=read_and_validate(out/'submission.csv',r['shapes'])
        if stage=='E054_CONTROL':assert r['csv_sha256']==r['config']['expected_visible_csv_sha256'],'Pure public control did not reproduce the expected visible CSV'
        save(done_path,dict(status='complete',kernel=launch['kernel'],version=launch['version'],csv_locally_validated=True,launch_receipt_sha256=sha(launch_path),result=r,counts={k:dict(nodes=len(n),edges=len(e)) for k,(n,e) in groups.items()}))
    elif action=='submit':
        done=read(done_path);assert done['status']=='complete' and done['csv_locally_validated'] and done['launch_receipt_sha256']==sha(launch_path)
        assert api.kernels_status(launch['kernel']).status.name=='COMPLETE'
        cfg=done['result']['config'];r=dict(status='request_pending',kernel=launch['kernel'],version=launch['version'],attempted_at_utc=datetime.now(timezone.utc).isoformat(),csv_sha256=done['result']['csv_sha256'],message='E054 joint frozen selection: '+cfg['selected']+'; full image inference, exploratory reused validation')
        if stage=='E054_CONTROL':r['message']='E054 pure public tight55 control; remove weighted bridges; visible CSV exactly reproduced, full hidden-test inference'
        with attempt_path.open('x') as handle:json.dump(r,handle,indent=2)
        response=api.competition_submit_code(file_name='submission.csv',message=r['message'],competition=COMP,kernel=launch['kernel'],kernel_version=launch['version'],quiet=True)
        r.update(status='submitted',ref=response.ref,response_message=response.message);save(attempt_path,r)
    elif action=='score':
        attempt=read(attempt_path);assert attempt['status']=='submitted'
        rows=api.competition_submissions(COMP,page_size=100);r=next(r for r in rows or [] if r is not None and str(r.ref)==str(attempt['ref']))
        save(Path(str(prefix)+'_status.json'),dict(ref=r.ref,status=getattr(r.status,'name',str(r.status)),public_score=r.public_score,error_description=r.error_description,checked_at_utc=datetime.now(timezone.utc).isoformat(),kernel=attempt['kernel'],version=attempt['version']))
    else:raise ValueError('Use check, submit or score')
if __name__=='__main__':main(*sys.argv[1:])
