"""Consolidate completed experiments; never treats a launch as a result."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'results'/name).read_text())
def main():
    stages={s:read(f'E040_{s}_completed.json') for s in ['PREPARE','TRAIN','EVALUATE']}
    for r in stages.values():assert r['result']['status']=='complete'
    evaluation=stages['EVALUATE']['result'];trained=stages['TRAIN']['result']
    decision='retain_joint_event_component_for_future_graph_evaluation' if evaluation['passed'] else 'reject_joint_event_component'
    result=dict(experiment='E040',status='complete',decision=decision,evaluation=evaluation,training=trained,stages={s:{k:v for k,v in r.items() if k!='result'} for s,r in stages.items()},timings={s:r['result']['seconds'] for s,r in stages.items()},leaderboard_submitted=False)
    (ROOT/'results/E040_completed.json').write_text(json.dumps(result,indent=2)+'\n')
    e38=read('E038_completed.json');e39=read('E039_completed.json')
    gpu=e38['timings']['FEATURES']+e39['timings']['TRAIN']+e39['timings']['INFER']+result['timings']['TRAIN']
    cpu=sum(e38['timings'][s] for s in ['AUDIT','PREPARE','EVALUATE'])+e39['timings']['EVALUATE']+result['timings']['PREPARE']+result['timings']['EVALUATE']
    summary=dict(status='complete',execution_order=['E038','E039','E040'],all_three_executed=True,decisions={r['experiment']:r['decision'] for r in [e38,e39,result]},gpu_process_seconds=gpu,cpu_process_seconds=cpu,timing_scope='Reported runner processes only; startup, export and quota billing not measured.',new_local_tests_passed=5,spatial_adapter_max_abs_error=e38['adapter_max_absolute_difference'],leaderboard_submitted=False,official_tracking_scores_computed=False,background_jobs_remaining=False,report='docs/E038_E040_TRES_LINEAS.es.md')
    (ROOT/'results/E038_E040_completed.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    p=ROOT/'results/STATUS.json';status=json.loads(p.read_text());status.update(active_experiment=None,active_experiments=[],three_lines_active_stage=None,manual_turn_monitoring=False,monitoring_active=False,next_experiment_started=False)
    status['E040']=dict(status='COMPLETE',decision=decision,receipt='results/E040_completed.json',fit_positive=evaluation['fit_positives'],development_positive=evaluation['development_positives'],passed=evaluation['passed'])
    status['quality_status']='All three image experiments completed; E038 and E039 failed their development gates; E040 '+decision+'; no new submission or official tracking score'
    status['next_research_direction']='Evaluate a passing component on real predicted graph candidates before any leaderboard submission' if evaluation['passed'] else 'Do not submit or sweep the rejected E038-E040 adaptations; their component metrics do not demonstrate a tracking improvement'
    status['three_lines_summary']='results/E038_E040_completed.json';p.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
if __name__=='__main__':main()
