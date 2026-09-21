"""Record the inspected public-source differences and completed CPU result."""
import hashlib,json
from pathlib import Path
refs={
 'zhehaoliang/biohub-p26-o01-exact-public-0947':'DeepCenter division threshold .20; our baseline .25.',
 'reyhanksatria/biohub-cell-tracking-0-947-lb':'DeepCenter division threshold .20; repackaged model paths.',
 'yudaiyamauchi/lb-exploration-c-disagreement-adaptive':'Confidence-margin-dependent primary/secondary blending; no independently verified improvement.',
 'yudaiyamauchi/lb-exploration-e-mutual-best':'Mutual row/column best bonus in logits before column softmax; E051 adapts the idea to cached probabilities.',
 'arnav170/biohub-reid3s':'Patch descriptors and link classifier; public default ReID weight zero. E051 uses independent simpler descriptors.',
 'noisyislands/biohub-transformer-finetune':'Frozen encoder with association-head fine tuning; synthetic negatives on partially labelled detections need review before adopting.'}
rows=[]
for ref,note in refs.items():
 p=Path('artifacts/research_latest_20260920')/ref.replace('/','__')/'source.py'
 rows.append(dict(ref=ref,url='https://www.kaggle.com/code/'+ref,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,finding=note))
Path('results/E051_public_audit.json').write_text(json.dumps(dict(date='2026-09-21',sources=rows,interpretation='Listed public scores acknowledged. Only actual source differences evaluated; titles and local proxy scores do not establish causality.'),indent=2)+'\n')
p=Path('results/STATUS.json');s=json.loads(p.read_text());s['quality_status']='E050 COMPLETE 0.946, no gain. E051 CPU alternatives below prior visual control. E052 public-config factorial replay in progress.'
s['E050'].update(submission_status='COMPLETE',public_score=.946,submission_ref=56409893)
s['E051']=dict(status='COMPLETE',promoted=False,receipt='results/E051_EVALUATE_completed.json',reason='All new confidence/mutual ensembles below visual-only control on eight reused videos')
s['E052']=dict(status='RUNNING',purpose='Frozen DeepCenter maps on GPU then public-threshold plus visual-association replay on CPU')
p.write_text(json.dumps(s,indent=2)+'\n')
p=Path('README.md');t=p.read_text(encoding='utf-8');t=t.replace('Primera consulta: pendiente, sin score todavía.','Resultado confirmado: **0.946**, sin mejora de leaderboard.');p.write_text(t,encoding='utf-8')
p=Path('docs/E049_E050_SUBMISSION.es.md');t=p.read_text(encoding='utf-8').replace('Primera consulta: PENDING, sin score público todavía.','Resultado confirmado el 2026-09-21: **COMPLETE, 0.946**, igual al control y la submission visual anterior.');p.write_text(t,encoding='utf-8')
