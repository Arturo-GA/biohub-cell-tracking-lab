"""Observable-confidence ensemble of captured Transformer, motion and appearance.

Inspired by public mutual-best, disagreement-adaptive and ReID notebooks.
This is a probability-level adaptation, not their pre-ILP logit implementation.
"""
import numpy as np
from scipy.spatial import cKDTree
from .visual_assignment import assign
SCALE=np.array([1.625,.40625,.40625])

def descriptors(nodes,image):
    ids=sorted(nodes);out={};grid=np.stack(np.meshgrid(*([np.arange(-3,4)]*3),indexing='ij'),axis=-1)
    radius=np.linalg.norm(grid,axis=-1);flatgrid=grid.reshape(-1,3)
    for t in sorted({n['t'] for n in nodes.values()}):
        frame=np.asarray(image[int(t),::1,::4,::4],np.float32)
        lo,hi=np.percentile(frame,[10,99.5]);frame=np.clip((frame-lo)/max(hi-lo,1e-6),0,2)
        padded=np.pad(frame,3,mode='edge')
        here=[k for k in ids if nodes[k]['t']==t]
        for start in range(0,len(here),256):
            batch=here[start:start+256];centers=np.rint([[nodes[k]['z'],nodes[k]['y']/4,nodes[k]['x']/4] for k in batch]).astype(int)
            centers=np.clip(centers,0,np.array(frame.shape)-1)+3
            at=centers[:,None,:]+flatgrid[None]
            patch=padded[at[:,:,0],at[:,:,1],at[:,:,2]]
            bg=np.median(patch[:,radius.ravel()>3],axis=1);positive=np.maximum(patch-bg[:,None],0)
            mass=np.maximum(positive.sum(1),1e-6);center=positive@flatgrid/mass[:,None]
            moment=np.sum(positive[:,:,None]*(flatgrid[None]-center[:,None])**2,axis=1)/mass[:,None]
            profile=np.stack([positive[:,(radius.ravel()>=r)&(radius.ravel()<r+1)].mean(1) for r in range(4)],axis=1)
            profile/=np.maximum(profile.max(1,keepdims=True),1e-6)
            desc=np.c_[positive.max(1),positive.mean(1),np.mean(positive>.25,axis=1),moment/9,profile]
            for k,d in zip(batch,desc):out[k]=d
    return out

def fuse(nodes,original,visual,appearance,mode):
    original=list(original);incoming={b:a for a,b in original};groups={};result={};stats=[]
    for e,p in visual.items():
        if e[0] in nodes and e[1] in nodes and nodes[e[1]]['t']==nodes[e[0]]['t']+1:
            groups.setdefault(nodes[e[0]]['t'],[]).append((e,float(p)))
    for t,items in sorted(groups.items()):
        es=[e for e,p in items];p=np.clip([p for e,p in items],1e-8,1)
        source=sorted({a for a,b in es});target=sorted({b for a,b in es});si={k:i for i,k in enumerate(source)};ti={k:i for i,k in enumerate(target)}
        a=np.array([si[x] for x,y in es]);b=np.array([ti[y] for x,y in es]);mass=np.bincount(b,weights=p,minlength=len(target))
        def normalized(logits):
            mx=np.full(len(target),-np.inf);np.maximum.at(mx,b,logits);v=np.exp(np.clip(logits-mx[b],-50,0));return v/np.maximum(np.bincount(b,weights=v,minlength=len(target))[b],1e-12)
        pv=p/np.maximum(mass[b],1e-12);q=pv.copy()
        if mode in ('adaptive','adaptive_mutual'):
            apos=np.array([[nodes[k][axis] for axis in ['z','y','x']] for k in source])*SCALE
            bpos=np.array([[nodes[k][axis] for axis in ['z','y','x']] for k in target])*SCALE
            velocity=np.zeros_like(apos);known=[]
            for i,k in enumerate(source):
                if k in incoming and nodes[k]['t']==nodes[incoming[k]]['t']+1:
                    prev=np.array([nodes[incoming[k]][axis] for axis in ['z','y','x']])*SCALE
                    velocity[i]=apos[i]-prev;known.append(i)
            if len(known)>=3:
                dist,near=cKDTree(apos[known]).query(apos,k=min(8,len(known)))
                for i in range(len(source)):
                    ix=np.asarray(known)[np.atleast_1d(near[i])[np.atleast_1d(dist[i])<=20]]
                    if len(ix)>=3:velocity[i]=.5*velocity[i]+.5*np.median(velocity[ix],axis=0)
            residual=np.linalg.norm(bpos[b]-apos[a]-velocity[a],axis=1)
            pg=normalized(-.5*(residual/3)**2)
            fa=np.array([appearance[k] for k in source]);fb=np.array([appearance[k] for k in target]);allf=np.r_[fa,fb]
            scale=np.maximum(np.percentile(allf,90,axis=0)-np.percentile(allf,10,axis=0),.1)
            delta=np.mean(np.minimum(np.abs(fa[a]-fb[b])/scale,3),axis=1)
            pa=normalized(-delta/.25-.5*(residual/6)**2)
            probs=np.stack([pv,pg,pa]);margins=[]
            for expert in probs:
                first=np.zeros(len(target));np.maximum.at(first,b,expert)
                arg=np.full(len(target),len(es),int);np.minimum.at(arg,b,np.where(expert==first[b],np.arange(len(es)),len(es)))
                second=np.zeros(len(target));np.maximum.at(second,b,np.where(np.arange(len(es))==arg[b],0,expert));margins.append(first-second)
            w=np.array([.7,.15,.15])[:,None]*(.05+np.array(margins));w/=w.sum(0)
            w[0]=np.maximum(w[0],.5);w[1:]*=(1-w[0])/np.maximum(w[1:].sum(0),1e-12)
            q=np.sum(w[:,b]*probs,axis=0);stats.append(float(w[0].mean()))
        if mode in ('mutual','adaptive_mutual'):
            bestsrc=np.zeros(len(source));besttgt=np.zeros(len(target));np.maximum.at(bestsrc,a,q);np.maximum.at(besttgt,b,q)
            row=q==bestsrc[a];col=q==besttgt[b];both=row&col
            bonus=.25*(col+.5*row+.5*both-.2*(~both))
            q=normalized(np.log(np.clip(q,1e-8,1))+bonus)
        result.update({e:float(np.clip(v*mass[j],1e-8,1)) for e,v,j in zip(es,q,b)})
    # Preserve any cached entries outside normal consecutive frames.
    result={**visual,**result}
    edges,report=assign(nodes,original,result)
    report.update(mean_transformer_weight=float(np.mean(stats)) if stats else 1.,mode=mode)
    return edges,report
