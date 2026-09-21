import hashlib,json
from pathlib import Path
from biohub_lab.submission import read_and_validate
paths={'public_v3':Path('artifacts/public_v3_outputs/submission.csv'),'public_exact0947':Path('artifacts/public_exact0947_outputs/submission.csv'),'our_tight55':Path('outputs/e050_submission/weighted_submission/public_tight55.csv'),'our_weighted':Path('outputs/e050_submission/submission.csv')}
shapes=json.loads(Path('outputs/e050_submission/weighted_submission/result.json').read_text())['shapes']
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};graphs={k:read_and_validate(p,shapes) for k,p in paths.items()}
assert hashes['public_v3']==hashes['our_tight55']
rows=[]
for label,g in graphs.items():
 for video,(nodes,edges) in g.items():
  bn,be=graphs['our_tight55'][video];ids=set(nodes);bi=set(bn)
  rows.append(dict(source=label,video=video,nodes=len(nodes),edges=len(edges),added_node_ids=len(ids-bi),removed_node_ids=len(bi-ids),changed_shared_coords=sum(nodes[k]!=bn[k] for k in ids&bi),added_edges=len(set(edges)-set(be)),removed_edges=len(set(be)-set(edges))))
result=dict(date='2026-09-21',sha256=hashes,public_v3_exactly_reproduced=True,scope='Visible test only; cannot infer private-test equality or attribute public score differences',rows=rows)
Path('results/E054_public_csv_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
