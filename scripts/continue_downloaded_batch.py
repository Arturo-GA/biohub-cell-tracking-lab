"""Finish one authorized local batch after its existing download process exits.

This is a finite job: no Kaggle status polling, scheduling or repeated launches.
"""
import argparse
from datetime import datetime,timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def validate_prepared_next(verified,next_folder,plan):
    metadata=json.loads((Path(next_folder)/'kernel-metadata.json').read_text())
    code=(Path(next_folder)/metadata['code_file']).resolve()
    if (not code.is_relative_to(Path(next_folder).resolve()) or
        hashlib.sha256(code.read_bytes()).hexdigest()!=plan['notebook_sha256'] or
        verified['next_offset']!=plan['offset']):
        raise ValueError('Prepared next batch does not match its verified plan')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pid',type=int,required=True)
    parser.add_argument('--download-created',type=float,required=True)
    parser.add_argument('--folder',required=True)
    parser.add_argument('--launch',required=True)
    parser.add_argument('--cpu-receipt',required=True)
    parser.add_argument('--local-receipt',required=True)
    parser.add_argument('--state',required=True)
    parser.add_argument('--next-folder',required=True)
    parser.add_argument('--next-receipt',required=True)
    parser.add_argument('--next-preflight',required=True)
    parser.add_argument('--kaggle-python',required=True)
    args=parser.parse_args();folder=Path(args.folder).resolve();state_path=Path(args.state).resolve()
    if not folder.is_relative_to(ROOT/'outputs') or not state_path.is_relative_to(ROOT/'outputs'):
        raise ValueError('Job inputs and state must stay in workspace outputs')
    if state_path.exists():raise ValueError('Job state already exists; preserve the previous attempt')
    state=dict(status='waiting_for_existing_download',pid=os.getpid(),
        process_created=psutil.Process().create_time(),started_at_utc=datetime.now(timezone.utc).isoformat(),
        folder=str(folder),download_pid=args.download_pid,download_created=args.download_created,
        automatic_kaggle_status_polling=False,training_started=False)
    def save(status,**fields):
        state.update(status=status,updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
        state_path.parent.mkdir(parents=True,exist_ok=True)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
        print('BATCH_JOB',status,flush=True)
    def run(command):subprocess.run(command,cwd=ROOT,check=True)
    save(state['status'])
    try:
        # Wait only for this already-running local transfer, never for a remote notebook.
        deadline=time.monotonic()+7200
        while True:
            try:
                process=psutil.Process(args.download_pid)
                same=abs(process.create_time()-args.download_created)<.01
                alive=same and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:alive=False
            if not alive:break
            if time.monotonic()>deadline:raise TimeoutError('Existing transfer exceeded the finite two-hour job limit')
            time.sleep(5)
        if not (folder/'download_verification.json').is_file():
            raise RuntimeError('Transfer ended without a complete verification receipt; no GPU job or upload started')
        save('verifying_cpu_inputs')
        audit=[sys.executable,'-u',str(ROOT/'scripts/audit_cpu_batch.py'),'--folder',str(folder),
               '--launch',args.launch,'--output',args.cpu_receipt]
        run(audit)
        plan=json.loads(Path(args.next_preflight).read_text())
        verified=json.loads(Path(args.cpu_receipt).read_text())
        validate_prepared_next(verified,args.next_folder,plan)
        save('launching_next_cpu_batch')
        try:
            run([args.kaggle_python,'-u',str(ROOT/'scripts/launch_cpu_notebook.py'),args.next_folder,
                 '--receipt',args.next_receipt])
            state['next_cpu_launch']='accepted';state['next_cpu_receipt']=args.next_receipt
        except subprocess.CalledProcessError as error:
            # A remote launch failure must not discard already verified local work.
            state['next_cpu_launch']='failed';state['next_cpu_launch_exit_code']=error.returncode
        save('processing_local_videos')
        run([sys.executable,'-u',str(ROOT/'scripts/run_local_prepared_batch.py'),
             '--cpu-root',str(folder/'local_inputs')])
        save('verifying_local_features')
        run(audit+['--local-output',args.local_receipt])
        save('complete',cpu_receipt=args.cpu_receipt,local_receipt=args.local_receipt,
             completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        save('failed',error=str(error));raise


if __name__=='__main__':main()
