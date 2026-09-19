"""Read the exact submitted candidate's competition status without resubmitting."""
import json
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    root=Path(__file__).resolve().parents[1]
    attempt=json.loads((root/'results/VISUAL_SUBMISSION_attempt.json').read_text())
    assert attempt['status']=='submitted' and attempt['ref']
    api=KaggleApi();api.authenticate()
    rows=api.competition_submissions('biohub-cell-tracking-during-development',page_size=100)
    row=next(r for r in rows or [] if r is not None and str(r.ref)==str(attempt['ref']))
    record=dict(ref=row.ref,status=getattr(row.status,'name',str(row.status)),
        public_score=row.public_score,error_description=row.error_description,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),kernel=attempt['kernel'],version=attempt['version'])
    (root/'results/VISUAL_SUBMISSION_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
