import hashlib,json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def main():
    receipt=json.loads((ROOT/'results/E036_launch.json').read_text());api=KaggleApi();api.authenticate()
    status=api.kernels_status(receipt['kernel']);print(status.status.name,flush=True)
    if status.status.name!='COMPLETE':return
    output=ROOT/'outputs/e036_joint';api.kernels_output(receipt['kernel'],str(output),file_pattern=r'.*(result.json|_metrics.json|_events.json)$',quiet=True)
    p=output/'joint_division/result.json';r=json.loads(p.read_text());assert r['status']=='complete' and r['config']==json.loads((ROOT/'baseline/e036_joint_division.json').read_text())
    summary=dict(status='complete',version=receipt['version'],kernel=receipt['kernel'],manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=r['seconds'],metrics=r['metrics'],changes=r['changes'],gpu=False,leaderboard_submitted=False)
    (ROOT/'results/E036_completed.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['metrics','changes','seconds']},indent=2))
if __name__=='__main__':main()
