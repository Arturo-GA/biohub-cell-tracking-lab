"""E038 CPU heldout localization probe: image pixels vs frozen pretrained features."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from association_cpu_runner import locate
def main(package):
    start=time.monotonic();out=Path('/kaggle/working/spatial_probe_evaluation');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e038_protocol.json').read_text());p=locate(Path('/kaggle/input'),'spatial_probe_features/result.json');m=json.loads(p.read_text());assert m['config']==cfg
    fit={k:[] for k in ['spatial','raw','offset']};dev=[]
    for r in m['videos']:
        path=p.parent/r['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with np.load(path) as d:a={k:d[k] for k in fit}
        if r['split']=='fit':
            for k in fit:fit[k].append(a[k])
        else:dev.append((r['video'],a))
    fit={k:np.concatenate(v) for k,v in fit.items()};models={}
    for key in ['raw','spatial']:
        models[key]=make_pipeline(StandardScaler(),Ridge(alpha=100.));models[key].fit(fit[key],fit['offset'])
    rows=[]
    for video,a in dev:
        row=dict(video=video,n=len(a['offset']))
        for key in ['zero','raw','spatial']:
            pred=np.zeros_like(a['offset']) if key=='zero' else np.clip(models[key].predict(a[key]),-4.5,4.5)
            err=np.linalg.norm(pred-a['offset'],axis=1)*1.625
            row[key]=dict(mean_um=float(err.mean()),median_um=float(np.median(err)),within3=int((err<=3).sum()))
        rows.append(row)
    means={key:float(np.mean([r[key]['mean_um'] for r in rows])) for key in ['zero','raw','spatial']}
    passing=means['spatial']<.9*min(means['zero'],means['raw']) and all(np.mean([r['spatial']['mean_um']-r['raw']['mean_um'] for r in rows if r['video'].startswith(e)])<0 for e in ['44b6','6bba'])
    result=dict(status='complete',config=cfg,mean_video_error_um=means,videos=rows,passed=bool(passing),seconds=time.monotonic()-start,scope='GT-seeded localization probe, not full detection or leaderboard score; 10% error reduction vs raw and zero, and positive direction in both embryos required.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('SPATIAL_RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
