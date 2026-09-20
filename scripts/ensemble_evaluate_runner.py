"""E044 deterministic rank-interleaved detector ensemble, CPU only."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from biohub_lab.temporal_detector import votes

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def one(relative):
    found=list(dict.fromkeys(p for depth in [1,2,3] for p in Path('/kaggle/input').glob('*/'*depth+relative)))
    assert len(found)==1,(relative,found)
    return found[0]
def matched(pred,gt,radius):
    if not len(pred) or not len(gt):return set()
    d=cdist(gt,pred)*1.625;penalty=(len(gt)+1)*(radius+1)
    a,b=linear_sum_assignment(np.c_[np.where(d<=radius,d,2*penalty),np.full((len(gt),len(gt)),penalty)])
    return {int(i) for i,j in zip(a,b) if j<len(pred) and d[i,j]<=radius}
def fuse(dense,static,budget=256):
    # Equal rank interleaving, dense first at tied rank; one isotropic voxel dedup.
    accepted=[];sources=[]
    for rank in range(max(len(dense),len(static))):
        for name,points in [('dense',dense),('static',static)]:
            if rank>=len(points):continue
            p=points[rank]
            if accepted and np.any(np.linalg.norm(np.asarray(accepted)-p,axis=1)<=1):continue
            accepted.append(p);sources.append(name)
            if len(accepted)==budget:return np.asarray(accepted),sources
    return np.asarray(accepted).reshape(-1,3),sources

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e044_protocol.json').read_text())
    manifests={}
    for name,item in cfg['sources'].items():
        p=one(item['path']);assert sha(p)==item['sha256'];manifests[name]=(p,json.loads(p.read_text()))
    dp,dm=manifests['dense'];sp,sm=manifests['static'];rp,rm=manifests['raw'];rows=[];diagnostics=[]
    assert {v['video'] for v in dm['videos']}=={v['video'] for v in sm['videos']}
    for r in dm['videos']:
        video=r['video'];s=next(v for v in sm['videos'] if v['video']==video);raw=next(v for v in rm['videos'] if v['video']==video)
        for path,digest in [(dp.parent/r['file'],r['sha256']),(sp.parent/s['file'],s['sha256']),(rp.parent/raw['truth'],raw['truth_sha256']),(rp.parent/raw['image'],raw['image_sha256'])]:assert sha(path)==digest
        with np.load(dp.parent/r['file']) as z:logits=z['logits']
        with np.load(sp.parent/s['file']) as z:fields=z['field_1'].astype(np.float32)
        with np.load(rp.parent/raw['truth']) as z:coords=z['coords']/[1,1,4,4]
        images=np.load(rp.parent/raw['image'],mmap_mode='r')
        for j,t in enumerate(cfg['frames']):
            v=logits[j];dense=np.argwhere(v==maximum_filter(v,3));dense=dense[np.argsort(-v[tuple(dense.T)],kind='stable')[:512]]
            static,_=votes(fields[j],np.asarray(images[t],np.float32));ensemble,sources=fuse(dense,static,cfg['budget']);gt=coords[coords[:,0]==t,1:]
            arms={'dense':dense[:256],'static':static[:256],'ensemble':ensemble,'union_diagnostic':np.concatenate([dense[:256],static[:256]])}
            for arm,points in arms.items():rows.append(dict(video=video,frame=t,arm=arm,predicted=len(points),gt=len(gt),matched3=len(matched(points,gt,3)),matched7=len(matched(points,gt,7))))
            for radius in [3,7]:
                a=matched(dense[:256],gt,radius);b=matched(static[:256],gt,radius)
                diagnostics.append(dict(video=video,frame=t,radius=radius,both=len(a&b),dense_only=len(a-b),static_only=len(b-a),neither=len(gt)-len(a|b),ensemble_dense=sources.count('dense'),ensemble_static=sources.count('static')))
        print('ENSEMBLE_VIDEO',video,flush=True)
    summary=[dict(embryo=e,arm=a,**{k:sum(r[k] for r in rows if r['arm']==a and (e=='all' or r['video'].startswith(e))) for k in ['gt','predicted','matched3','matched7']}) for e in ['all','44b6','6bba'] for a in arms]
    def get(e,a):return next(r for r in summary if r['embryo']==e and r['arm']==a)
    assert (get('all','dense')['matched3'],get('all','dense')['matched7'])==(227,321)
    assert (get('all','static')['matched3'],get('all','static')['matched7'])==(177,354)
    passed=all(get(e,'ensemble')['matched3']>=max(get(e,a)['matched3'] for a in ['dense','static']) and get(e,'ensemble')['matched7']>max(get(e,a)['matched7'] for a in ['dense','static']) for e in ['44b6','6bba'])
    out=Path('/kaggle/working/ensemble_evaluation');out.mkdir(exist_ok=True)
    result=dict(status='complete',config=cfg,summary=summary,rows=rows,diagnostics=diagnostics,passed=passed,seconds=time.monotonic()-start,scope='Repeated sparse development recall; union has doubled budget and is diagnostic only; no graph or leaderboard score.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('ENSEMBLE_RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
