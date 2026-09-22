"""Bounded, foreground continuation of the authorized E063 experiment (no submission)."""
import json,os,subprocess,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
def run(args,log):
    with log.open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1'),stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    start=time.monotonic();api=KaggleApi();api.authenticate()
    for stage in ['DETECT','PREPARE','INFER','SELECT','EVALUATE','AUDIT']:
        name='E063_'+stage;receipt=ROOT/f'results/{name}_launch.json'
        if not receipt.exists():
            assert stage!='DETECT','Start reviewed neural detection explicitly first'
            run(['scripts/build_trajectory_selector.py',stage],ROOT/f'results/{name}_build.txt')
            run(['scripts/audit_new_payloads.py','kaggle/'+name.lower()],ROOT/f'results/{name}_audit.txt')
            args=['scripts/launch_three_lines_gpu.py',name] if stage=='INFER' else ['scripts/launch_cpu_notebook.py','kaggle/'+name.lower(),'--receipt','results/'+name+'_launch.json']
            run(args,ROOT/f'results/{name}_push.txt');print('LAUNCHED',name,flush=True)
        r=json.loads(receipt.read_text());previous=None
        while True:
            if time.monotonic()-start>7200:raise TimeoutError('E063 continuation deadline')
            status=api.kernels_status(r['kernel']).status.name
            if status!=previous:print(name,status,flush=True);previous=status
            if status=='COMPLETE':break
            if status not in ('RUNNING','QUEUED'):raise RuntimeError(name+' '+status)
            time.sleep(30)
        completed=ROOT/f'results/{name}_completed.json'
        if not completed.exists():run(['scripts/check_three_lines.py',name],ROOT/f'results/{name}_check.txt')
        assert completed.exists()
        result=json.loads(completed.read_text())['result'];print('VERIFIED',name,'seconds',result.get('seconds'),'selected',result.get('selected'),flush=True)
    print('E063_ALL_COMPLETE_NO_SUBMISSION',flush=True)
if __name__=='__main__':main()
