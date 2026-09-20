"""Foreground, one-shot execution of the user's E041 -> E042 -> E043 batch.

No scheduler, background service or submission. Stop on runtime errors so they
can be inspected; scientific gate failures do not stop the next experiment.
"""
import json,subprocess,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parents[1]
STAGES=['E041_PREPARE','E041_REGRID','E041_TRAIN','E041_EVALUATE','E042_PREPARE','E042_TRAIN','E042_EVALUATE','E043_EVALUATE']

def command(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def status(stage,state):
    p=ROOT/'results/STATUS.json';d=json.loads(p.read_text());experiment=stage.split('_')[0]
    d.update(active_experiment=experiment,active_experiments=[experiment],three_lines_active_stage=stage,manual_turn_monitoring=True)
    d[experiment].update(status=state,stage=stage,leaderboard_submitted=False);p.write_text(json.dumps(d,indent=2)+'\n')

def main():
    api=KaggleApi();api.authenticate()
    for stage in STAGES:
        complete=ROOT/f'results/{stage}_completed.json';receipt=ROOT/f'results/{stage}_launch.json'
        if complete.exists():continue
        if not receipt.exists():
            command('scripts/build_next_three.py',stage)
            if stage.endswith('_TRAIN'):command('scripts/launch_three_lines_gpu.py',stage)
            else:command('scripts/launch_cpu_notebook.py','kaggle/'+stage.lower(),'--receipt',str(receipt))
        r=json.loads(receipt.read_text());status(stage,'RUNNING');last=None
        while True:
            current=api.kernels_status(r['kernel']).status.name
            if current!=last:print('BATCH_STAGE',stage,current,flush=True);last=current
            if current=='COMPLETE':break
            if current in ['ERROR','CANCELLED','CANCELED']:
                status(stage,'RUNTIME_ERROR');raise RuntimeError(stage+' '+current+'; inspect before retry')
            time.sleep(45)
        command('scripts/check_three_lines.py',stage)
        assert complete.exists()
        if stage.endswith('_EVALUATE'):
            d=json.loads(complete.read_text());(ROOT/f'results/{stage[:4]}_completed.json').write_text(json.dumps(d,indent=2)+'\n')
            status(stage,'COMPLETE');print('BATCH_RESULT',stage,json.dumps(d['result'].get('passed')),flush=True)
    p=ROOT/'results/STATUS.json';d=json.loads(p.read_text());d.update(active_experiment=None,active_experiments=[],three_lines_active_stage=None,manual_turn_monitoring=False,quality_status='E041-E043 complete; inspect component and full-graph gates; no new submission')
    p.write_text(json.dumps(d,indent=2)+'\n')
    result={e:json.loads((ROOT/f'results/{e}_completed.json').read_text()) for e in ['E041','E042','E043']}
    (ROOT/'results/E041_E043_completed.json').write_text(json.dumps(dict(status='complete',experiments=result,leaderboard_submitted=False),indent=2)+'\n');print('BATCH_COMPLETE',flush=True)

if __name__=='__main__':main()
