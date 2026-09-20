"""One status/download operation, with no polling or implicit launch."""
import hashlib,json,sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def main():
    stage=sys.argv[1];receipt=json.loads((ROOT/f'results/{stage}_launch.json').read_text());api=KaggleApi();api.authenticate()
    status=api.kernels_status(receipt['kernel']).status.name;print(stage,status,flush=True)
    if status!='COMPLETE':return
    out=ROOT/'outputs'/stage.lower();pattern=r'.*(result.json|metrics.json|development_scores.npz)$'
    if stage.endswith('_TRAIN'):pattern=r'.*(result.json|\.pt)$'
    api.kernels_output(receipt['kernel'],str(out),file_pattern=pattern,quiet=True)
    paths=list(out.rglob('result.json'));assert len(paths)==1;result=json.loads(paths[0].read_text());assert result['status']=='complete'
    if stage.endswith('_TRAIN'):
        weights=result.get('models',[])
        if 'checkpoint' in result:weights=weights+[dict(checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'])]
        assert weights
        for item in weights:assert hashlib.sha256((paths[0].parent/item['checkpoint']).read_bytes()).hexdigest()==item['sha256']
    record=dict(kernel=receipt['kernel'],version=receipt['version'],manifest_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),result=result)
    (ROOT/f'results/{stage}_completed.json').write_text(json.dumps(record,indent=2)+'\n')
    brief={k:v for k,v in result.items() if k not in ['config','videos','rows','outputs','history','reports']}
    if 'videos' in result:brief['videos_count']=len(result['videos'])
    print(json.dumps(brief,indent=2),flush=True)
if __name__=='__main__':main()
