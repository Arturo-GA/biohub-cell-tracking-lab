"""Submit the already-produced pure public-matched E050 ablation once."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
from biohub_lab.submission import read_and_validate
ROOT=Path(__file__).resolve().parents[1]
def main():
    launch=json.loads((ROOT/'results/E050_SUBMISSION_launch.json').read_text())
    done=json.loads((ROOT/'results/E050_SUBMISSION_completed.json').read_text());r=done['result'];assert done['status']=='complete'
    public=ROOT/'artifacts/public_v3_outputs/submission.csv';digest=hashlib.sha256(public.read_bytes()).hexdigest()
    assert digest==r['public_control_sha256'];groups=read_and_validate(public,r['shapes'])
    api=KaggleApi();api.authenticate();assert api.kernels_status(launch['kernel']).status.name=='COMPLETE'
    record=dict(status='request_pending',kernel=launch['kernel'],version=launch['version'],file_name='weighted_submission/public_tight55.csv',attempted_at_utc=datetime.now(timezone.utc).isoformat(),csv_sha256=digest,counts={k:dict(nodes=len(n),edges=len(e)) for k,(n,e) in groups.items()},public_reference='raunakdey07/biohub-harmonic-fusion-v3',public_visible_csv_identical=True,uses_our_recomputed_predictions=True,message='Pure public tight55 ablation; remove E050 complementary bridges; full hidden-test inference from our notebook')
    path=ROOT/'results/E050_CONTROL_attempt.json'
    if '--diagnose-rejection' in sys.argv:
        previous=json.loads(path.read_text());assert previous['status']=='request_pending'
        rows=api.competition_submissions('biohub-cell-tracking-during-development',page_size=100)
        assert not any(r and r.description==record['message'] for r in rows),'Submission already exists'
        record['prior_http_400_confirmed_no_submission']=True
    else:
        with path.open('x') as f:json.dump(record,f,indent=2)
    try:
        response=api.competition_submit_code(file_name=record['file_name'],message=record['message'],competition='biohub-cell-tracking-during-development',kernel=launch['kernel'],kernel_version=launch['version'],quiet=True)
    except Exception as exc:
        response=getattr(exc,'response',None);record.update(status='rejected' if response is not None and response.status_code==400 else 'uncertain',error_type=type(exc).__name__,error_body=response.text[:4000] if response is not None else str(exc));path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));raise
    record.update(status='submitted',ref=response.ref,response_message=response.message);path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
