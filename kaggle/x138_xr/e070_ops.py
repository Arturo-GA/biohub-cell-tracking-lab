"""Receipt-backed E070 launches, bounded waits, downloads, and verification.

No leaderboard submission. The thread heartbeat resumes downstream work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import threading
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'results/E070'; RES.mkdir(exist_ok=True)
LEDGER=RES/'ledger.json'


def now():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,r):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,default=str)+'\n',encoding='utf-8');tmp.replace(p)
def state():return load(LEDGER) if LEDGER.exists() else dict(experiment='E070',target_ready=5,reset_utc='2026-09-28T00:00:00+00:00',runs={},ready_existing=['C3_fork8'],auto_submit=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check(api):
    ledger=state();changes=[]
    for name,row in ledger['runs'].items():
        if not row.get('version'):continue
        status=api.kernels_status(row['kernel']).status.name
        if status!=row.get('status'):changes.append(dict(name=name,old=row.get('status'),status=status))
        row['status']=status
    ledger['checked_at_utc']=now();save(LEDGER,ledger)
    print(json.dumps(dict(checked_at_utc=ledger['checked_at_utc'],changes=changes,runs={k:v.get('status') for k,v in ledger['runs'].items()})),flush=True)
    return ledger,changes


def launch(api,name):
    lock=RES/(name+'.launch.lock')
    with lock.open('x') as f:f.write(now())
    try:
        receipt=RES/(name+'_launch.json')
        assert not receipt.exists(),'Already launched or attempted; inspect receipt, do not duplicate'
        folder=ROOT/'kaggle/x138_xr'/('k_e070_'+name.lower())
        metadata=load(folder/'kernel-metadata.json')
        assert metadata['is_private'] is True
        assert not metadata['enable_internet']
        resp=api.kernels_push(str(folder));r=resp.to_dict()
        row=dict(kernel=metadata['id'],version=r.get('versionNumber'),response=r,
                 folder=str(folder),gpu=metadata['enable_gpu'],notebook_sha256=sha(folder/metadata['code_file']),
                 time_utc=now(),status='LAUNCHED' if r.get('versionNumber') and not r.get('error') else 'LAUNCH_FAILED',leaderboard_submitted=False)
        save(receipt,row)
        ledger=state();ledger['runs'][name]=row;save(LEDGER,ledger)
        print(json.dumps(row),flush=True)
        assert row['status']=='LAUNCHED',r
    finally:lock.unlink(missing_ok=True)


def download(api,name):
    receipt=RES/(name+'_launch.json');r=load(receipt)
    assert api.kernels_status(r['kernel']).status.name=='COMPLETE'
    folder=Path(r['folder']);assert sha(folder/'notebook.ipynb')==r['notebook_sha256']
    out=ROOT/'outputs/e070'/name
    is_lab=name.startswith('lab_')
    pattern=r'(e070_.*|\.log$)' if is_lab else r'(submission\.csv$|run_manifest\.json$|\.log$)'
    files,_=api.kernels_output(r['kernel'],str(out),file_pattern=pattern,quiet=True)
    if is_lab:
        work=out/('e070_'+name[-1]);actual=load(work/'run_manifest.json');expected=load(folder/'expected_manifest.json')
        assert actual==expected
        complete=load(work/'complete.json');assert complete['videos']==199 and complete['configurations']==6
        summary=dict(kernel=r['kernel'],version=r['version'],manifest_verified=True,complete=complete,files=len(files))
        save(RES/(name+'_completed.json'),summary)
    else:
        logs=list(out.glob('*.log'));assert len(logs)==1,[p.name for p in logs]
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'kaggle/x138_xr/verify_run.py'),
                        str(folder),str(out),str(receipt),str(logs[0]),'--save',str(RES/(name+'_verified.json'))],check=True)
        summary=load(RES/(name+'_verified.json'))
    ledger=state();ledger['runs'][name]['status']='COMPLETE';ledger['runs'][name]['download_verified']=True;save(LEDGER,ledger)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','check','wait','download','logs']);p.add_argument('name',nargs='?');p.add_argument('--seconds',type=int,default=50);a=p.parse_args()
    api=KaggleApi();api.authenticate()
    if a.action=='launch':launch(api,a.name)
    elif a.action=='download':download(api,a.name)
    elif a.action=='check':check(api)
    elif a.action=='logs':
        targets=state()['runs'];out=ROOT/'outputs/e070/streams';out.mkdir(parents=True,exist_ok=True)
        def stream(name,row):
            path=out/(name+'.txt');old=set(path.read_text(encoding='utf-8').splitlines()) if path.exists() else set()
            try:
                with path.open('w',encoding='utf-8') as f:
                    for event in api.kernels_logs_stream(row['kernel']):
                        chunk=event.get('data','');f.write(chunk);f.flush()
                        for line in chunk.splitlines():
                            if line not in old and any(k in line for k in ('E070','Traceback','AssertionError','RuntimeError','REPAIR FAILED','GATE','Submission written','final rows')):
                                print(name,line[-3000:],flush=True)
            except Exception as exc:print(name,'LOG STREAM ERROR',type(exc).__name__,str(exc)[:250],flush=True)
        threads=[]
        for name,row in targets.items():
            if row.get('version') and (not a.name or name==a.name):
                t=threading.Thread(target=stream,args=(name,row),daemon=True);t.start();threads.append(t)
        deadline=time.monotonic()+a.seconds
        while any(t.is_alive() for t in threads) and time.monotonic()<deadline:time.sleep(.5)
    else:
        deadline=time.monotonic()+a.seconds
        while True:
            ledger,changes=check(api)
            terminal=[x for x in changes if x['status'] in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELED')]
            if terminal or all(r.get('status') in ('COMPLETE','ERROR','LAUNCH_FAILED') for r in ledger['runs'].values()) or time.monotonic()>=deadline:break
            time.sleep(min(30,max(0,deadline-time.monotonic())))


if __name__=='__main__':main()
