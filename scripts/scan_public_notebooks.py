import json
from pathlib import Path
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();results=[]
for sort in ['scoreDescending','dateRun','dateCreated']:
    items=api.kernels_list(competition='biohub-cell-tracking-during-development',sort_by=sort,page_size=40)
    records=[]
    for item in items:
        keys=['ref','title','last_run_time','total_votes','current_version_number']+[k for k in dir(item) if 'score' in k.lower() and not k.startswith('_')]
        records.append({k:getattr(item,k,None) for k in dict.fromkeys(keys) if not callable(getattr(item,k,None))})
    results.append(dict(sort=sort,records=records))
    print(sort,json.dumps(records[:15],default=str),flush=True)
Path('results/public_notebooks_20260920_latest.json').write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),lists=results),default=str,indent=2)+'\n')
