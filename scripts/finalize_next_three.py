"""Audit completed E041-E043 receipts and record resource/solver evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads((ROOT/p).read_text())

def main():
    stages=['E041_PREPARE','E041_REGRID','E041_TRAIN','E041_EVALUATE','E042_PREPARE','E042_TRAIN','E042_EVALUATE','E043_EVALUATE']
    receipts=[];gpu=cpu=0.
    for stage in stages:
        launch=read(f'results/{stage}_launch.json');complete=read(f'results/{stage}_completed.json');folder=ROOT/'kaggle'/stage.lower();meta=json.loads((folder/'kernel-metadata.json').read_text())
        assert complete['version']==launch['version'] and complete['result']['status']=='complete'
        assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==launch['notebook_sha256']
        assert meta['is_private'] and not meta['enable_tpu'] and meta['enable_gpu']==stage.endswith('_TRAIN')
        seconds=complete['result']['seconds']
        if meta['enable_gpu']:gpu+=seconds
        else:cpu+=seconds
        receipts.append(dict(stage=stage,kernel=launch['kernel'],version=launch['version'],gpu=meta['enable_gpu'],seconds=seconds,notebook_sha256=launch['notebook_sha256']))
    graph=read('results/E043_completed.json')['result'];frames=[f for r in graph['reports'] for f in r['frames']]
    solver=dict(frames=len(frames),map_optimal=sum(f['map_optimal'] for f in frames),consensus_optimal=sum(f['consensus_optimal'] for f in frames),sample_optimal=sum(f['sample_optimal'] for f in frames),samples_attempted=3*len(frames),uncertain_events=sum(f['uncertain_events'] for f in frames),original_forks=sum(f['original_forks'] for f in frames),map_forks=sum(f['map_forks'] for f in frames),consensus_forks=sum(f['consensus_forks'] for f in frames),changed_edges={a:sum(r['changed_edges'][a] for r in graph['reports']) for a in ['map','consensus']})
    result=dict(status='complete',stages=receipts,gpu_process_seconds=gpu,cpu_process_seconds=cpu,resource_scope='Final successful stages only. Excludes startup/export, failed preparation v1, and superseded regrid v1. Not billed Kaggle quota.',solver=solver,weight_verification=read('results/E041_E043_weight_verification.json'),local_checks=read('results/E041_E043_local_checks.json'),leaderboard_submitted=False)
    (ROOT/'results/E041_E043_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    status=read('results/STATUS.json')
    for e in ['E041','E042','E043']:status[e].update(status='COMPLETE',receipt=f'results/{e}_completed.json',passed=read(f'results/{e}_completed.json')['result']['passed'],leaderboard_submitted=False)
    status['E041']['decision']='No promotion: better tight localization but lower 7um recall than static field. Retain dense-supervision signal.'
    status['E042']['decision']='No native-resolution promotion: slightly worse in both embryos. Query permutation shows dependence on temporal reference.'
    status['E043']['decision']='Review candidate before submission' if any(graph['passed'].values()) else 'No promotion: full-graph gate not met; do not tune on these same outcomes.'
    status.update(active_experiment=None,active_experiments=[],three_lines_active_stage=None,manual_turn_monitoring=False,three_lines_summary='results/E041_E043_completed.json',quality_status='E041-E043 completed sequentially; E041/E042 gates failed with component signals; E043 '+str(graph['passed'])+'; no new submission')
    (ROOT/'results/STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['stages','weight_verification','local_checks']},indent=2))

if __name__=='__main__':main()
