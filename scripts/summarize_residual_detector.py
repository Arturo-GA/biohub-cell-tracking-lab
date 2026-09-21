"""Durable decision from completed E061/E062, without claiming leaderboard gains."""
import json
from pathlib import Path

def result(stage):return json.loads(Path('results/'+stage+'_completed.json').read_text())['result']

rows=[]
for stage in ('E061_EVALUATE','E062_EVALUATE'):
    r=result(stage);base=r['metrics']['control']
    for arm,m in r['metrics'].items():
        rows.append(dict(stage=stage,arm=arm,score=m['score'],delta=m['score']-base['score'],
            edge_tp_delta=m['edge_tp']-base['edge_tp'],edge_fp_delta=m['edge_fp']-base['edge_fp'],
            added_nodes=sum(x.get('added_nodes',0) for x in r['reports'] if x['arm']==arm),
            added_edges=sum(x.get('added_edges',0) for x in r['reports'] if x['arm']==arm),
            passed=arm in r['passed']))
stages=('E061_PREPARE','E061_TRAIN','E061_INFER','E061_EVALUATE','E061_COVERAGE','E062_PREPARE','E062_INFER','E062_EVALUATE')
durations={s:result(s)['seconds'] for s in stages}
summary=dict(variants=rows,coverage=result('E061_COVERAGE')['totals'],seconds=durations,
    gpu_process_seconds=sum(durations[s] for s in ('E061_TRAIN','E061_INFER','E062_INFER')),
    cpu_process_seconds=sum(durations[s] for s in stages if s not in ('E061_TRAIN','E061_INFER','E062_INFER')),
    fit_videos=40,evaluation_videos=24,fit_evaluation_disjoint=True,independent_project_holdout=False,
    gpu_quota_billing_seconds_unknown=True,leaderboard_submitted=False)
Path('results/E061_E062_decision.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(variants=rows,gpu_process_seconds=summary['gpu_process_seconds'],cpu_process_seconds=summary['cpu_process_seconds']),indent=2))
