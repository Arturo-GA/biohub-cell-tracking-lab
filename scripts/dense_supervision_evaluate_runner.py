"""E041 CPU equal-budget sparse recall; no unlabeled false-positive claims."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from spatial_probe_features_runner import one,sha
from temporal_detector_evaluate_runner import matches

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/dense_supervision_evaluation');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e041_protocol.json').read_text());p=one('dense_supervision_training/result.json');m=json.loads(p.read_text());assert m['config']==cfg
    q=one('three_lines_data/result.json');raw=json.loads(q.read_text());c=one('temporal_detector_evaluation/result.json');control=json.loads(c.read_text());rows=[]
    for r in m['videos']:
        path=p.parent/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as d:logits=d['logits']
        record=next(a for a in raw['videos'] if a['video']==r['video'])
        with np.load(q.parent/record['truth']) as d:coords=d['coords']/[1,1,4,4]
        for j,t in enumerate(cfg['frames']):
            v=logits[j];xyz=np.argwhere(v==maximum_filter(v,3));xyz=xyz[np.argsort(-v[tuple(xyz.T)],kind='stable')[:512]];gt=coords[coords[:,0]==t,1:]
            for budget in [128,256,512]:rows.append(dict(video=r['video'],frame=t,arm='dense',budget=budget,gt=len(gt),matched3=matches(xyz[:budget],gt,3),matched7=matches(xyz[:budget],gt,7)))
    summary=[]
    for embryo in ['all','44b6','6bba']:
        for arm in ['dense','image_peaks','static','temporal']:
            for budget in [128,256,512]:
                rr=[r for r in rows+control['rows'] if r['arm']==arm and r['budget']==budget and (embryo=='all' or r['video'].startswith(embryo))]
                summary.append(dict(embryo=embryo,arm=arm,budget=budget,**{k:sum(r[k] for r in rr) for k in ['gt','matched3','matched7']}))
    def get(e,a):return next(r for r in summary if r['embryo']==e and r['arm']==a and r['budget']==256)
    passed=all(get(e,'dense')['matched3']>max(get(e,a)['matched3'] for a in ['image_peaks','static','temporal']) and get(e,'dense')['matched7']>=max(get(e,a)['matched7'] for a in ['image_peaks','static','temporal']) for e in ['44b6','6bba'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',summary=summary,rows=rows,passed=bool(passed),seconds=time.monotonic()-start,scope='Sparse recall only; repeated development; no official graph score'),indent=2));print('DENSE_RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
