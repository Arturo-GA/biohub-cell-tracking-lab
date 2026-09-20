"""E040 CPU: geometry baseline and genuine event precision gate."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from association_cpu_runner import locate

def operating_point(y,scores):
    for value in np.unique(scores)[::-1]:
        selected=scores>=value;tp=int(y[selected].sum());fp=int(selected.sum()-tp)
        if fp:return None
        if tp>=3:return dict(threshold=float(value),tp=tp,fp=fp)
    return None

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/expanded_event_evaluation');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e040_protocol.json').read_text());inputs=Path('/kaggle/input');p=locate(inputs,'expanded_event_data/result.json');m=json.loads(p.read_text());q=locate(inputs,'expanded_event_training/result.json');trained=json.loads(q.read_text());assert trained['config']==m['config']==cfg
    pin=json.loads((package/'baseline/e040_input_pin.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['manifest_sha256']
    xf=[];yf=[];dev=[]
    for r in m['videos']:
        if r['empty']:continue
        path=p.parent/r['events'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['events_sha256']
        with np.load(path) as d:x=d['geometry'];y=d['y']
        if r['split']=='fit':xf.append(x);yf.append(y)
        else:
            scores=next(a for a in trained['outputs'] if a['video']==r['video']);sp=q.parent/scores['file'];assert hashlib.sha256(sp.read_bytes()).hexdigest()==scores['sha256']
            with np.load(sp) as d:s={k:d[k] for k in ['temporal','repeated_center']}
            dev.append((r['video'],x,y,s))
    model=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.8,class_weight='balanced',random_state=cfg['seed'],n_jobs=2).fit(np.concatenate(xf),np.concatenate(yf))
    y=np.concatenate([r[2] for r in dev]);scores={k:np.concatenate([r[3][k] for r in dev]) for k in ['temporal','repeated_center']};scores['geometry']=np.concatenate([model.predict_proba(r[1])[:,1] for r in dev]);arms={}
    for k,s in scores.items():arms[k]=dict(ap=float(average_precision_score(y,s)),operating_point=operating_point(y,s))
    passed=arms['temporal']['operating_point'] is not None and arms['temporal']['ap']>=arms['geometry']['ap']+.05 and arms['temporal']['ap']>=arms['repeated_center']['ap']+.05
    names=np.concatenate([np.repeat(r[0],len(r[2])) for r in dev]);by_embryo={}
    for embryo in ['44b6','6bba']:
        mask=np.char.startswith(names,embryo);by_embryo[embryo]=dict(positive=int(y[mask].sum()),negative=int((y[mask]==0).sum()),ap={k:float(average_precision_score(y[mask],s[mask])) for k,s in scores.items()})
    result=dict(status='complete',config=cfg,fit_positives=int(np.concatenate(yf).sum()),development_positives=int(y.sum()),development_negatives=int((y==0).sum()),arms=arms,by_embryo=by_embryo,passed=bool(passed),seconds=time.monotonic()-start,scope='Annotated candidate event classification only. No changes to tracking or leaderboard. Development threshold needs3TP/0knownFP; AP improvement0.05 vs geometry and repeated-center image control. Cohort division enriched.')
    (out/'result.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'development_scores.npz',y=y,video=names,**scores);print('EVENT_RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
