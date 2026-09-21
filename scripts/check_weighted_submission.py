"""Download and validate the exact finished notebook output; never submit here."""
import hashlib,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from biohub_lab.submission import read_and_validate

ROOT=Path(__file__).resolve().parents[1]
def main():
    launch=json.loads((ROOT/'results/E050_SUBMISSION_launch.json').read_text())
    api=KaggleApi();api.authenticate();state=api.kernels_status(launch['kernel']).status.name
    print(state,flush=True)
    if state!='COMPLETE':return
    out=ROOT/'outputs/e050_submission'
    api.kernels_output(launch['kernel'],str(out),file_pattern=r'(^submission.csv$|^weighted_submission/result.json$)',quiet=True)
    result=json.loads((out/'weighted_submission/result.json').read_text());assert result['status']=='complete'
    csv=out/'submission.csv';assert hashlib.sha256(csv.read_bytes()).hexdigest()==result['csv_sha256']
    groups=read_and_validate(csv,result['shapes'])
    record=dict(kernel=launch['kernel'],version=launch['version'],status='complete',result=result,
        csv_locally_validated=True,counts={k:dict(nodes=len(n),edges=len(e)) for k,(n,e) in groups.items()},
        launch_receipt_sha256=hashlib.sha256((ROOT/'results/E050_SUBMISSION_launch.json').read_bytes()).hexdigest())
    (ROOT/'results/E050_SUBMISSION_completed.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
