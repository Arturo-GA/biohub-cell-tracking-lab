"""Foreground E067 continuation, including the explicitly requested submission."""
import json,os,subprocess,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
def run(args,label):
    with (ROOT/f'results/{label}_progress.txt').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=str(ROOT/'src')),stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    start=time.monotonic();api=KaggleApi();api.authenticate()
    for stage in ('EVALUATE','SUBMISSION'):
        name='E067_'+stage;p=ROOT/f'results/{name}_launch.json'
        if not p.exists():
            assert stage=='SUBMISSION'
            run(['scripts/build_trajectory_denoise.py',stage],name+'_build');run(['scripts/audit_new_payloads.py','kaggle/'+name.lower()],name+'_audit');run(['scripts/launch_three_lines_gpu.py',name],name+'_push');print('LAUNCHED',name,flush=True)
        r=json.loads(p.read_text());previous=None
        while True:
            if time.monotonic()-start>14400:raise TimeoutError('E067 four-hour foreground limit')
            state=api.kernels_status(r['kernel']).status.name
            if state!=previous:print(name,state,flush=True);previous=state
            if state=='COMPLETE':break
            if state not in ('QUEUED','RUNNING'):raise RuntimeError(name+' '+state)
            time.sleep(30)
        if not (ROOT/f'results/{name}_completed.json').exists():
            run(['scripts/check_three_lines.py',name] if stage=='EVALUATE' else ['scripts/denoise_submission_ops.py','check'],name+'_check')
        done=json.loads((ROOT/f'results/{name}_completed.json').read_text());print('VERIFIED',name,done['result'].get('selected',done['result'].get('config',{}).get('selected')),flush=True)
    attempt=ROOT/'results/E067_SUBMISSION_attempt.json'
    if not attempt.exists():run(['scripts/denoise_submission_ops.py','submit'],'E067_SUBMISSION_submit')
    r=json.loads(attempt.read_text());assert r['status']=='submitted',r['status'];print('SUBMITTED',r['ref'],flush=True)
    run(['scripts/denoise_submission_ops.py','score'],'E067_SUBMISSION_score')
if __name__=='__main__':main()
