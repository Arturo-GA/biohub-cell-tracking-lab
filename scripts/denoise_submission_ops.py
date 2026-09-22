"""Validate E067, submit once with a durable receipt, and read the result."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
from biohub_lab.submission import read_and_validate
ROOT=Path(__file__).resolve().parents[1];COMP='biohub-cell-tracking-during-development'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,indent=2),flush=True)
def main(action):
    api=KaggleApi();api.authenticate();lp=ROOT/'results/E067_SUBMISSION_launch.json';launch=json.loads(lp.read_text());done=ROOT/'results/E067_SUBMISSION_completed.json';attempt=ROOT/'results/E067_SUBMISSION_attempt.json'
    if action=='check':
        state=api.kernels_status(launch['kernel']).status.name;print(state,flush=True)
        if state!='COMPLETE':return
        out=ROOT/'outputs/e067_submission';api.kernels_output(launch['kernel'],str(out),file_pattern=r'(^submission.csv$|^trajectory_denoise_submission/(result.json|control.csv)$)',quiet=True)
        r=json.loads((out/'trajectory_denoise_submission/result.json').read_text());assert r['status']=='complete' and sha(out/'submission.csv')==r['csv_sha256'] and sha(out/'trajectory_denoise_submission/control.csv')==r['control_sha256']
        groups=read_and_validate(out/'submission.csv',r['shapes']);base=read_and_validate(out/'trajectory_denoise_submission/control.csv',r['shapes'])
        assert all(groups[v][1]==base[v][1] and set(groups[v][0])==set(base[v][0]) for v in groups)
        changed=sum(groups[v][0][k]!=base[v][0][k] for v in groups for k in groups[v][0]);assert changed>0 and changed==sum(x['moved'] for x in r['reports'])
        reference=json.loads((ROOT/'results/E054_CONTROL_completed.json').read_text())['result']['csv_sha256']
        save(done,dict(status='complete',kernel=launch['kernel'],version=launch['version'],csv_locally_validated=True,launch_receipt_sha256=sha(lp),changed_nodes=changed,visible_baseline_reproduced=r['control_sha256']==reference,result=r))
    elif action=='submit':
        d=json.loads(done.read_text());assert d['csv_locally_validated'] and d['launch_receipt_sha256']==sha(lp) and d['changed_nodes']>0
        assert api.kernels_status(launch['kernel']).status.name=='COMPLETE'
        message='E067 robust full-track acceleration denoising; exploratory new localization, unchanged lineage topology'
        r=dict(status='request_pending',kernel=launch['kernel'],version=launch['version'],attempted_at_utc=datetime.now(timezone.utc).isoformat(),csv_sha256=d['result']['csv_sha256'],message=message)
        with attempt.open('x') as f:json.dump(r,f,indent=2)
        response=api.competition_submit_code(file_name='submission.csv',message=message,competition=COMP,kernel=launch['kernel'],kernel_version=launch['version'],quiet=True)
        r.update(status='submitted',ref=response.ref,response_message=response.message);save(attempt,r)
    elif action=='score':
        a=json.loads(attempt.read_text());assert a['status']=='submitted';rows=api.competition_submissions(COMP,page_size=100);r=next(r for r in rows if r and str(r.ref)==str(a['ref']))
        save(ROOT/'results/E067_SUBMISSION_status.json',dict(ref=r.ref,status=getattr(r.status,'name',str(r.status)),public_score=r.public_score,error_description=r.error_description,checked_at_utc=datetime.now(timezone.utc).isoformat()))
    else:raise ValueError(action)
if __name__=='__main__':main(sys.argv[1])
