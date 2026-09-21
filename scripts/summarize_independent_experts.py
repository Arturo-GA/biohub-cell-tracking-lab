"""Report completed E059 evidence without mistaking oracle unions for ensembles."""
import json,sys
from pathlib import Path

stage=sys.argv[1] if len(sys.argv)>1 else 'E059'
assert stage in ('E059','E060')
root=Path('outputs')/(stage.lower()+'_evaluate')/('independent_evaluation' if stage=='E059' else 'selective_independent')
r=json.loads((root/'result.json').read_text())
base=json.loads((root/'visual_metrics.json').read_text())
baseline={s['dataset']:s for s in base['samples']}
rows=[]
for arm,metric in r['metrics'].items():
    samples=json.loads((root/(arm+'_metrics.json')).read_text())['samples']
    rows.append(dict(arm=arm,score=metric['score'],delta=metric['score']-r['metrics']['visual']['score'],
        edge_tp=metric['edge_tp'],edge_fp=metric['edge_fp'],
        **r['complementarity'][arm],
        videos=[dict(video=s['dataset'],delta=s['adj_edge_jaccard']-baseline[s['dataset']]['adj_edge_jaccard'],
            tp_delta=s['edge_tp']-baseline[s['dataset']]['edge_tp'],
            fp_delta=s['edge_fp']-baseline[s['dataset']]['edge_fp']) for s in samples]))
result=dict(arms=rows,gpu_seconds=json.loads(Path('results/E059_CAPTURE_completed.json').read_text())['result']['seconds'],
    cpu_seconds=r['seconds'],scope=r['scope'],leaderboard_submitted=False,
    warning='Rescued links are diagnostic, not an oracle-selection submission score.')
Path('results/'+stage+'_decision.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
