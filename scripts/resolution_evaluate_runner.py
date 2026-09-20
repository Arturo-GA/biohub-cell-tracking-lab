import json,sys,time
from pathlib import Path
import numpy as np
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/resolution_evaluation');out.mkdir(exist_ok=True);p=one('resolution_training/result.json');m=json.loads(p.read_text());rows=[]
    for arm in m['models']:
        for r in arm['outputs']:
            q=p.parent/r['file'];assert sha(q)==r['sha256']
            with np.load(q) as d:err=np.linalg.norm(d['pred']-d['target'],axis=1)*1.625;zero=np.linalg.norm(d['target'],axis=1)*1.625;shuffled=np.linalg.norm(d['pred_shuffled']-d['target'],axis=1)*1.625
            rows.append(dict(arm=arm['arm'],video=r['video'],samples=len(err),mean_error_um=float(err.mean()),zero_error_um=float(zero.mean())))
            rows.append(dict(arm=arm['arm']+'_shuffled_query',video=r['video'],samples=len(err),mean_error_um=float(shuffled.mean()),zero_error_um=float(zero.mean())))
    summary=[]
    for e in ['all','44b6','6bba']:
        for arm in ['coarse','native','coarse_shuffled_query','native_shuffled_query']:
            r=[a for a in rows if a['arm']==arm and (e=='all' or a['video'].startswith(e))];summary.append(dict(embryo=e,arm=arm,error=float(np.mean([a['mean_error_um'] for a in r])),zero=float(np.mean([a['zero_error_um'] for a in r]))))
    def get(e,a):return next(r['error'] for r in summary if r['embryo']==e and r['arm']==a)
    passed=get('all','native')<.9*get('all','coarse') and all(get(e,'native')<get(e,'coarse') for e in ['44b6','6bba'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',rows=rows,summary=summary,passed=bool(passed),seconds=time.monotonic()-start,scope='GT-seeded localization probe, not whole-video detection or tracking'),indent=2));print('RESOLUTION_RESULT',summary,passed,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
