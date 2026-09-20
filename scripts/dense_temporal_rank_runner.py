"""E032: temporal parent ranking on real detections, cross-embryo head validation.

Frozen detector features; this trains a small association head, not the UNet.
Unknown detections are not labelled background. Parent negatives are accepted
only outside the true parent's spatial matching tolerance.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from biohub_lab.identity_parent import parent_table
from association_cpu_runner import locate
SCALE=np.array([1.625,.40625,.40625])

def match(coords,truth):
    mapping={}
    for t in np.unique(truth[:,1]):
        a=np.flatnonzero(truth[:,1]==t);b=np.flatnonzero(coords[:,0]==t)
        if not len(b):continue
        d=cdist(truth[a,2:]*SCALE,coords[b,1:]*SCALE);penalty=(len(a)+1)*8
        cost=np.c_[np.where(d<=7,d,2*penalty),np.full((len(a),len(a)),penalty)]
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if j<len(b) and d[i,j]<=7:mapping[int(truth[a[i],0])]=int(b[j])
    return mapping

def examples(coords,features,truth,edges):
    mapping=match(coords,truth);gt={int(r[0]):r for r in truth};parents=parent_table(coords,12,20.)
    features=features.astype(np.float32);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
    X=[];Y=[];M=[];counts=dict(gt_edges=len(edges),represented=0,ambiguous=0)
    for a,b in edges:
        if a not in mapping or b not in mapping:continue
        source,target=mapping[a],mapping[b];ps=parents[target];valid=ps>=0;correct=np.flatnonzero(ps==source)
        if not len(correct):continue
        counts['represented']+=1
        if valid.sum()<2:continue
        counts['ambiguous']+=1;safe=np.maximum(ps,0)
        delta=(coords[safe,1:]-coords[target,1:])*SCALE/20
        appearance=np.c_[np.abs(features[safe]-features[target]),features[safe]*features[target]]
        x=np.c_[appearance,delta,np.linalg.norm(delta,axis=1)]
        # Other detections near the annotated parent may be duplicate localization
        # hypotheses, so their relation label is left unknown during training.
        far=np.linalg.norm((coords[safe,1:]-gt[a][2:])*SCALE,axis=1)>7.
        mask=valid & (far | (ps==source))
        X.append(x);Y.append(int(correct[0]));M.append(np.stack([valid,mask]))
    return np.asarray(X,np.float32).reshape(-1,12,132),np.asarray(Y,np.int64),np.asarray(M,bool).reshape(-1,2,12),counts

class Ranker(nn.Module):
    def __init__(self,image=True):
        super().__init__();self.image=image
        self.net=nn.Sequential(nn.Linear(132 if image else 4,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x if self.image else x[...,-4:]).squeeze(-1)

def main(package):
    start=time.monotonic();torch.set_num_threads(2);root=Path('/kaggle/working/dense_temporal_rank');root.mkdir(exist_ok=True)
    inputs=Path('/kaggle/input');config=json.loads((package/'baseline/e032_temporal_rank.json').read_text())
    source=locate(inputs,'dense_detector_predictions/result.json');manifest=json.loads(source.read_text());assert manifest['status']=='complete'
    prep=locate(inputs,'dense_detector_inputs/result.json').parent;dataset={};counts={}
    for r in manifest['files']:
        if r['mode']!=config['mode']:continue
        p=source.parent/r['file'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==r['sha256']
        with np.load(p,allow_pickle=False) as d:
            truth=json.loads((prep/(r['video']+'_truth.json')).read_text())
            x,y,m,c=examples(d['coords'],d['features'],np.asarray(truth['nodes'],float),truth['edges'])
        dataset[r['video']]=(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(m));counts[r['video']]=c
        print('TEMPORAL_DATA',r['video'],c,flush=True)
    results=[]
    for heldout_embryo in ['44b6','6bba']:
        train=[v for n,v in dataset.items() if not n.startswith(heldout_embryo) and len(v[1])]
        test={n:v for n,v in dataset.items() if n.startswith(heldout_embryo)}
        assert train and test
        for image in [False,True]:
            torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed']);model=Ranker(image)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for step in range(config['steps']):
                x,y,m=train[int(rng.integers(len(train)))];ix=rng.integers(len(y),size=128)
                pred=model(x[ix]).masked_fill(~m[ix,1],-1e4);loss=nn.functional.cross_entropy(pred,y[ix]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();opt.step()
            reports=[];model.eval()
            for name,(x,y,m) in test.items():
                correct=nearest=shuffled=0
                with torch.inference_mode():
                    for first in range(0,len(y),256):
                        z=x[first:first+256];mask=m[first:first+256,0];target=y[first:first+256]
                        pred=model(z).masked_fill(~mask,-1e4);correct+=int((pred.argmax(1)==target).sum())
                        nearest+=int((z[:,:,-1].masked_fill(~mask,1e4).argmin(1)==target).sum())
                        scrambled=z.clone();scrambled[...,:128]=torch.roll(z[...,:128],1,0)
                        shuffled+=int((model(scrambled).masked_fill(~mask,-1e4).argmax(1)==target).sum())
                reports.append(dict(video=name,targets=len(y),correct=correct,nearest=nearest,shuffled=shuffled))
            path=root/(heldout_embryo+('_image.pt' if image else '_geometry.pt'))
            torch.save(dict(state_dict=model.state_dict(),image=image,heldout_embryo=heldout_embryo,config=config),path)
            report=dict(heldout_embryo=heldout_embryo,image=image,steps=config['steps'],videos=reports,targets=sum(r['targets'] for r in reports),correct=sum(r['correct'] for r in reports),nearest=sum(r['nearest'] for r in reports),shuffled=sum(r['shuffled'] for r in reports),weight_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
            results.append(report);print('TEMPORAL_HEAD',json.dumps(report),flush=True)
    result=dict(status='complete',config=config,counts=counts,results=results,seconds=time.monotonic()-start,scope='Cross-embryo association-head validation, conditional on detector provenance and represented GT edges; not full graph score',leaderboard_submitted=False,encoder_trained=False,training_device='CPU')
    (root/'result.json').write_text(json.dumps(result,indent=2));print('TEMPORAL_RANK_COMPLETE',flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
