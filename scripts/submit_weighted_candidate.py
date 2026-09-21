"""One authorized exploratory leaderboard submission, with durable attempt ID."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    root=Path(__file__).resolve().parents[1]
    launch=json.loads((root/'results/E050_SUBMISSION_launch.json').read_text())
    done=json.loads((root/'results/E050_SUBMISSION_completed.json').read_text())
    assert done['status']=='complete' and done['csv_locally_validated']
    assert done['launch_receipt_sha256']==hashlib.sha256((root/'results/E050_SUBMISSION_launch.json').read_bytes()).hexdigest()
    api=KaggleApi();api.authenticate();assert api.kernels_status(launch['kernel']).status.name=='COMPLETE'
    record=dict(status='request_pending',kernel=launch['kernel'],version=launch['version'],
        attempted_at_utc=datetime.now(timezone.utc).isoformat(),csv_sha256=done['result']['csv_sha256'],
        message='Public tight55 plus dense-static reliability-weighted gap completion; exploratory E049/E050')
    target=root/'results/E050_SUBMISSION_attempt.json'
    with target.open('x') as handle:json.dump(record,handle,indent=2)
    response=api.competition_submit_code(file_name='submission.csv',message=record['message'],competition='biohub-cell-tracking-during-development',kernel=launch['kernel'],kernel_version=launch['version'],quiet=True)
    record.update(status='submitted',ref=response.ref,response_message=response.message)
    target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
