import json,subprocess
from pathlib import Path
import numpy as np
from biohub_lab.temporal_detector import votes
old={};exec(subprocess.check_output(['git','show','b28fa94:src/biohub_lab/temporal_detector.py'],text=True),old)
rng=np.random.default_rng(47);image=rng.random((12,12,12),dtype=np.float32);field=rng.uniform(-2,2,(3,12,12,12)).astype(np.float32)
a,b=votes(field,image);c,d=old['votes'](field,image);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
Path('results/E045_vote_regression.json').write_text(json.dumps(dict(seed=47,old_revision='b28fa94',identical_coordinates=True,identical_scores=True),indent=2)+'\n')
print('E039 voting unchanged after exposing full score map: PASS')
for stage in ['E045','E046']:
    p=Path(f'results/{stage}_EVALUATE_completed.json');Path(f'results/{stage}_completed.json').write_text(p.read_text())
counts={}
for p in Path('outputs/e046_evaluate/graph_complements').glob('*_metrics.json'):
    r=json.loads(p.read_text());counts[p.stem.removesuffix('_metrics')]={k:sum(s[k] for s in r['samples']) for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']}
Path('results/E046_counts_audit.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)
p=Path('results/STATUS.json');s=json.loads(p.read_text())
for stage in ['E045','E046']:
    r=json.loads(Path(f'results/{stage}_completed.json').read_text())['result'];s[stage]=dict(status='COMPLETE',passed=r['passed'],seconds=r['seconds'],leaderboard_submitted=False)
s['E047']=dict(status='CPU_EVALUATION_RUNNING',gpu_seconds=68.501324461,trained=False,leaderboard_submitted=False);p.write_text(json.dumps(s,indent=2)+'\n')
