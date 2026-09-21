"""E045 six complementary detector designs evaluated on cached predictions, CPU only."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
from ensemble_evaluate_runner import one,sha,matched
from biohub_lab.temporal_detector import votes
from biohub_lab.detector_complements import candidates

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e045_protocol.json').read_text());sources={}
    for name,item in cfg['sources'].items():
        p=one(item['path']);assert sha(p)==item['sha256'];sources[name]=(p,json.loads(p.read_text()))
    dp,dm=sources['dense'];sp,sm=sources['static'];rp,rm=sources['raw'];rows=[]
    for r in dm['videos']:
        video=r['video'];s=next(v for v in sm['videos'] if v['video']==video);raw=next(v for v in rm['videos'] if v['video']==video)
        for path,digest in [(dp.parent/r['file'],r['sha256']),(sp.parent/s['file'],s['sha256']),(rp.parent/raw['truth'],raw['truth_sha256']),(rp.parent/raw['image'],raw['image_sha256'])]:assert sha(path)==digest
        with np.load(dp.parent/r['file']) as z:logits=z['logits']
        with np.load(sp.parent/s['file']) as z:fields={i:z['field_'+str(i)].astype(np.float32) for i in [1,3]}
        images=np.load(rp.parent/raw['image'],mmap_mode='r')
        # Predictions are fixed without access to annotations.
        predicted=[]
        for j,t in enumerate(cfg['frames']):
            image=np.asarray(images[t],np.float32)
            static,ss=votes_map(fields[1][j],image);temporal,ts=votes_map(fields[3][j],image)
            dog=gaussian_filter(image,.8)-gaussian_filter(image,2);ip=np.argwhere((dog==maximum_filter(dog,3))&(dog>0));ip=ip[np.argsort(-dog[tuple(ip.T)],kind='stable')[:512]]
            for budget in cfg['budgets']:predicted.append((t,budget,candidates(logits[j],ss,ts,ip,static,temporal,budget)))
        with np.load(rp.parent/raw['truth']) as z:coords=z['coords']/[1,1,4,4]
        for t,budget,arms in predicted:
            gt=coords[coords[:,0]==t,1:]
            for arm,points in arms.items():
                assert len(points)<=budget and len(np.unique(points,axis=0))==len(points)
                rows.append(dict(video=video,frame=t,arm=arm,budget=budget,predicted=len(points),gt=len(gt),matched3=len(matched(points,gt,3)),matched7=len(matched(points,gt,7))))
        print('ENSEMBLE_VIDEO',video,flush=True)
    summary=[dict(embryo=e,arm=a,budget=b,**{k:sum(r[k] for r in rows if r['arm']==a and r['budget']==b and (e=='all' or r['video'].startswith(e))) for k in ['gt','predicted','matched3','matched7']}) for e in ['all','44b6','6bba'] for b in cfg['budgets'] for a in arms]
    def get(e,a,b=256):return next(r for r in summary if r['embryo']==e and r['arm']==a and r['budget']==b)
    for a,expected in [('dense',(227,321)),('static',(177,354)),('temporal',(175,344)),('image_peaks',(194,319))]:assert (get('all',a)['matched3'],get('all',a)['matched7'])==expected
    controls=['dense','static','temporal','image_peaks']
    passed={a:all(get(e,a)['matched3']>=max(get(e,c)['matched3'] for c in controls) and get(e,a)['matched7']>max(get(e,c)['matched7'] for c in controls) for e in ['44b6','6bba']) for a in arms if a not in controls}
    out=Path('/kaggle/working/complements_evaluation');out.mkdir(exist_ok=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,summary=summary,rows=rows,passed=passed,seconds=time.monotonic()-start,scope='Six prespecified designs, repeated sparse development. No leaderboard or full graph claim.'),indent=2))
    print('ENSEMBLE_RESULT',json.dumps([s for s in summary if s['embryo']=='all']),json.dumps(passed),flush=True)

def votes_map(field,image):
    # Keep the identical E039 voting computation but return its entire map.
    from biohub_lab.temporal_detector import vote_map
    score=vote_map(field,image)
    xyz=np.argwhere((score==maximum_filter(score,3))&(score>0));xyz=xyz[np.argsort(-score[tuple(xyz.T)],kind='stable')[:512]]
    return xyz,score
if __name__=='__main__':main(Path(sys.argv[1]))
