"""Local CPU duplicate check only; never exports a submission."""
import json,time,hashlib
from pathlib import Path
from biohub_lab.submission import read_and_validate
from biohub_lab.trajectory_denoise import ARMS,refine
start=time.monotonic();p=Path('outputs/e064_submission/native_centers_submission/control.csv');receipt=json.loads(Path('results/E064_SUBMISSION_completed.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['result']['control_sha256'];shapes=receipt['result']['shapes'];graphs=read_and_validate(p,shapes);reports={a:dict(moved=0,videos=[]) for a in ARMS}
for v,(nodes,edges) in graphs.items():
    for a,options in ARMS.items():
        ns,r=refine(nodes,edges,**options,shape=shapes[v]);reports[a]['moved']+=r['moved'];reports[a]['videos'].append(dict(video=v,**r))
r=dict(status='complete',reports=reports,seconds=time.monotonic()-start,control_sha256=receipt['result']['control_sha256'],scope='Local CPU visible-test duplicate check only. No labels or submission CSV exported.')
Path('results/E067_PREVIEW_completed.json').write_text(json.dumps(r,indent=2)+'\n');print({a:reports[a]['moved'] for a in reports})
