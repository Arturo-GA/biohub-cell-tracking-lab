"""Process a completed CPU batch sequentially using the existing local guards."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def packages(root):
    root=Path(root).resolve()
    record=json.loads((root/'batch_result.json').read_text())
    split=json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']
    ordered=split['fit']+split['calibration']
    names=record['videos'];offset=record['offset'];end=record['next_offset']
    if (record['status']!='complete' or record['completed']!=names or not 1<=len(names)<=4 or
        not 0<=offset<end<=len(ordered) or record['source_bytes']>record['budget_bytes'] or
        names!=ordered[offset:end] or record['annotations_read'] or record['accelerator']!='none'):
        raise ValueError('Incomplete or inconsistent fixed-cohort CPU batch')
    result=[]
    for name in names:
        folder=(root/name).resolve()
        if not folder.is_relative_to(root):raise ValueError('Unsafe input path')
        child=json.loads((folder/'result.json').read_text())
        if child['status']!='complete' or child['video']!=name or child['offset']!=ordered.index(name):
            raise ValueError('Child input receipt does not match the batch')
        result.append(folder)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-root',required=True)
    parser.add_argument('--run-state')
    args=parser.parse_args()
    folders=packages(args.cpu_root)
    state_path=Path(args.run_state) if args.run_state else Path(args.cpu_root).parent/'local_run.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status')=='running':
        raise RuntimeError('Existing running batch receipt; inspect it before starting another process')
    state=dict(status='running',pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),
               videos=[p.name for p in folders],completed=[],training_started=False)
    def save():
        state_path.parent.mkdir(parents=True,exist_ok=True)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    save()
    try:
        for folder in folders:
            state['current']=folder.name;save()
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_local_prepared_video.py'),
                            '--cpu-package',str(folder)],cwd=ROOT,check=True)
            state['completed'].append(folder.name);save()
        state.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise
