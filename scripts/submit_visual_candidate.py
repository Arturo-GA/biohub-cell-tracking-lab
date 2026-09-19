"""Submit the validated private notebook once; preserve ambiguous attempts."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    def read(name):return json.loads((root/'results'/name).read_text())
    validation=read('E023_completed.json');completed=read('VISUAL_SUBMISSION_completed.json')
    launch=read('VISUAL_SUBMISSION_launch.json')
    assert validation['promote'] and all(validation['promotion_checks'].values())
    assert completed['status']=='complete' and completed['csv_locally_validated']
    assert completed['launch_receipt_sha256']==hashlib.sha256((root/'results/VISUAL_SUBMISSION_launch.json').read_bytes()).hexdigest()
    assert launch['kernel']=='jarturo/biohub-lab-visual-global-assignment'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    state=api.kernels_status(launch['kernel'])
    assert getattr(state.status,'name',str(state.status))=='COMPLETE'
    target=root/'results/VISUAL_SUBMISSION_attempt.json'
    record=dict(status='request_pending',kernel=launch['kernel'],version=launch['version'],
        attempted_at_utc=datetime.now(timezone.utc).isoformat(),csv_sha256=completed['csv_sha256'],
        message='Exact visual global assignment; frozen reserved validation E023')
    # Exclusive creation prevents automatic retries after uncertain API responses.
    with target.open('x') as handle:json.dump(record,handle,indent=2)
    response=api.competition_submit_code(file_name='submission.csv',message=record['message'],
        competition='biohub-cell-tracking-during-development',kernel=launch['kernel'],
        kernel_version=launch['version'],quiet=True)
    record.update(status='submitted',ref=response.ref,response_message=response.message)
    target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
