"""Read this exact submission's status and public score without resubmitting."""
import json
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    root=Path(__file__).resolve().parents[1]
    attempt=json.loads((root/'results/E050_SUBMISSION_attempt.json').read_text())
    assert attempt['status']=='submitted' and attempt['ref']
    api=KaggleApi();api.authenticate()
    rows=api.competition_submissions('biohub-cell-tracking-during-development',page_size=100)
    r=next(r for r in rows or [] if r is not None and str(r.ref)==str(attempt['ref']))
    record=dict(ref=r.ref,status=getattr(r.status,'name',str(r.status)),public_score=r.public_score,
        error_description=r.error_description,checked_at_utc=datetime.now(timezone.utc).isoformat(),kernel=attempt['kernel'],version=attempt['version'])
    (root/'results/E050_SUBMISSION_status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
