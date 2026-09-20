"""E039 CPU center voting, equal-budget one-to-one sparse recall, no false-positive claims."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from association_cpu_runner import locate
from biohub_lab.temporal_detector import votes

def matches(pred,gt,radius):
    if not len(pred) or not len(gt):return 0
    d=cdist(gt,pred)*1.625;penalty=(len(gt)+1)*(radius+1)
    a,b=linear_sum_assignment(np.c_[np.where(d<=radius,d,2*penalty),np.full((len(gt),len(gt)),penalty)])
    return sum(int(j<len(pred) and d[i,j]<=radius) for i,j in zip(a,b))

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/temporal_detector_evaluation');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e039_protocol.json').read_text());inputs=Path('/kaggle/input')
    p=locate(inputs,'three_lines_data/result.json');m=json.loads(p.read_text());q=locate(inputs,'temporal_detector_fields/result.json');fields=json.loads(q.read_text());assert fields['config']==cfg;rows=[]
    for item in fields['videos']:
        raw=next(r for r in m['videos'] if r['video']==item['video']);vol=np.load(p.parent/raw['image'],mmap_mode='r')
        with np.load(p.parent/raw['truth']) as d:coords=d['coords']/[1,1,4,4]
        field=q.parent/item['file'];assert hashlib.sha256(field.read_bytes()).hexdigest()==item['sha256']
        with np.load(field) as d:flow={c:d[f'field_{c}'] for c in [1,3]}
        for j,t in enumerate(cfg['frames']):
            gt=coords[coords[:,0]==t,1:];image=np.array(vol[t],np.float32)
            dog=gaussian_filter(image,.8)-gaussian_filter(image,2);xyz=np.argwhere((dog==maximum_filter(dog,3))&(dog>0));xyz=xyz[np.argsort(-dog[tuple(xyz.T)],kind='stable')[:512]]
            preds={'image_peaks':xyz,'static':votes(flow[1][j].astype(np.float32),image)[0],'temporal':votes(flow[3][j].astype(np.float32),image)[0]}
            for arm,points in preds.items():
                for budget in cfg['budgets']:
                    rows.append(dict(video=item['video'],frame=t,arm=arm,budget=budget,predicted=min(budget,len(points)),gt=len(gt),matched3=matches(points[:budget],gt,3),matched7=matches(points[:budget],gt,7)))
        print('CENTER_METRIC',item['video'],flush=True)
    summary=[]
    for embryo in ['all','44b6','6bba']:
        for arm in ['image_peaks','static','temporal']:
            for budget in cfg['budgets']:
                r=[r for r in rows if r['arm']==arm and r['budget']==budget and (embryo=='all' or r['video'].startswith(embryo))]
                summary.append(dict(embryo=embryo,arm=arm,budget=budget,gt=sum(a['gt'] for a in r),matched3=sum(a['matched3'] for a in r),matched7=sum(a['matched7'] for a in r)))
    def get(e,a):return next(r for r in summary if r['embryo']==e and r['arm']==a and r['budget']==256)
    passed=all(get(e,'temporal')['matched3']>max(get(e,'static')['matched3'],get(e,'image_peaks')['matched3']) and get(e,'temporal')['matched7']>=max(get(e,'static')['matched7'],get(e,'image_peaks')['matched7']) for e in ['44b6','6bba'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,summary=summary,rows=rows,passed=bool(passed),seconds=time.monotonic()-start,scope='Sparse center recall at equal budgets, not precision or official tracking score. Preregistered gate at256; other budgets are diagnostics only.'),indent=2));print('CENTER_RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
