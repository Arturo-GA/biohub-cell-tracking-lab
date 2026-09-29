"""Explicit one-shot E069 operations. Does not schedule or submit predictions."""
import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/E069'

def save(name,data):
    (RESULT/name).write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(data,default=str),flush=True)

def response_dict(r):return r.to_dict() if hasattr(r,'to_dict') else r.__dict__

def main(action):
    api=KaggleApi();api.authenticate()
    if action=='check':
        rows=[]
        for file in ('duplicates_launch.json','image_cache_launch.json','local_eval_launch.json'):
            if not (RESULT/file).exists():continue
            r=json.loads((RESULT/file).read_text());s=api.kernels_status(r['kernel'])
            rows.append(dict(kernel=r['kernel'],status=s.status.name,failure=getattr(s,'failure_message','')))
        save('status.json',dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),runs=rows))
    elif action in ('download_images','download_duplicates','download_evaluation'):
        stage={'download_images':'image_cache','download_duplicates':'duplicates','download_evaluation':'local_eval'}[action]
        r=json.loads((RESULT/(stage+'_launch.json')).read_text())
        state=api.kernels_status(r['kernel']).status.name
        assert state=='COMPLETE',(r['kernel'],state)
        output=ROOT/'outputs/e069'/stage
        files,_=api.kernels_output(r['kernel'],str(output),file_pattern=r'(e069_.*|\.log$)',quiet=True)
        mode={'image_cache':'image_cache','duplicates':'duplicates','local_eval':'local_linker_eval'}[stage]
        folder={'image_cache':'k_e069_images','duplicates':'k_e069_duplicates','local_eval':'k_e069_local_cpu'}[stage]
        local=ROOT/'kaggle/x138_xr'/folder
        assert hashlib.sha256((local/'notebook.ipynb').read_bytes()).hexdigest()==r['notebook_sha256']
        expected=json.loads((local/'expected_manifest.json').read_text())
        actual=json.loads((output/('e069_'+mode)/'run_manifest.json').read_text())
        assert expected==actual
        summary=json.loads((output/('e069_'+mode)/'complete.json').read_text())
        save(stage+'_completed.json',dict(kernel=r['kernel'],version=r['version'],files=len(files),summary=summary,manifest_verified=True))
    elif action=='upload_local':
        folder=ROOT/'outputs/e069/local_cache'
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['videos']==32
        panel=json.loads((folder/'panel.json').read_text())
        expected=json.loads((ROOT/'kaggle/x138_xr/k_e069_images/expected_manifest.json').read_text())
        assert json.loads((folder/'e069_local_manifest.json').read_text())['image_manifest']==expected
        for row in panel:
            with gzip.open(folder/(row['stem']+'.json.gz'),'rt',encoding='utf-8') as f:r=json.load(f)
            assert r['stem']==row['stem'] and r['checkpoint_sha256']==expected['checkpoint_sha256']
            assert r.get('joins_scored'), 'Fragment-rescue inference is not complete'
            assert r.get('forks_scored'), 'Independent division evidence is not complete'
        assert not (RESULT/'local_dataset_created.json').exists(), 'Dataset already created; do not duplicate upload'
        upload=ROOT/'outputs/e069/local_upload';upload.mkdir(exist_ok=True)
        for name in ('panel.json','complete.json','e069_local_manifest.json'):shutil.copyfile(folder/name,upload/name)
        records={}
        for row in panel:
            with gzip.open(folder/(row['stem']+'.json.gz'),'rt',encoding='utf-8') as f:records[row['stem']]=json.load(f)
        with gzip.open(upload/'e069_records.bin','wt',encoding='utf-8') as f:json.dump(records,f)
        metadata=dict(title='Biohub E069 local linker cache',id='jarturo/biohub-e069-local-linker-cache',
                      licenses=[{'name':'CC0-1.0'}],description='Private experiment: learned pair logits on fixed c3 detections. No images, annotations, credentials, or weights.')
        (upload/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        allowed={'e069_records.bin','e069_local_manifest.json','panel.json','complete.json','dataset-metadata.json'}
        assert {p.name for p in upload.iterdir()}==allowed
        save('local_dataset_attempt.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),private=True,files=sorted(allowed)))
        r=response_dict(api.dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False))
        assert not r.get('error'),r
        save('local_dataset_created.json',r)
    elif action=='launch_local_eval':
        receipt=RESULT/'local_eval_launch.json'
        assert not receipt.exists(),'Evaluation already launched'
        print('dataset status:',api.dataset_status('jarturo/biohub-e069-local-linker-cache'),flush=True)
        folder=ROOT/'kaggle/x138_xr/k_e069_local_cpu'
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        r=response_dict(api.kernels_push(str(folder)))
        assert not r.get('error') and r.get('versionNumber'),r
        save(receipt.name,dict(kernel=metadata['id'],version=r['versionNumber'],response=r,gpu=False,is_private=True,
             notebook_sha256=hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest(),
             time_utc=datetime.now(timezone.utc).isoformat(),leaderboard_submitted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','download_images','download_duplicates','download_evaluation','upload_local','launch_local_eval']);main(p.parse_args().action)
