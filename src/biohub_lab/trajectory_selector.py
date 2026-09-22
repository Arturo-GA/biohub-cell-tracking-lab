"""Learn trajectory quality from observable features, never annotation density."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.optimize import minimize
from biohub_lab.residual_tracklets import SCALE

FEATURE_NAMES=['log_fresh_length','anchors','left_anchor','right_anchor']+[
    signal+'_'+stat for signal in ('detector_confidence','edge_probability','competition_margin','speed_um','acceleration_um','distance_to_original_um','neighbors_within_7um')
    for stat in ('min','median','mean','max','std')]

def usefulness_label(links,known,correct,already):
    """None means unobserved; redundant correct links are not useful additions."""
    values=[known[e] for e in links if e in known]
    if not values:return None
    novel=any(e in correct and correct[e] not in already for e in links)
    return int(all(values) and novel)

def proposals(coords,mapping,confidence,paths,edges,prob):
    """All novel chain segments before confidence/acceleration filtering."""
    scores={tuple(e):float(p) for e,p in zip(edges,prob)}
    row={};col={}
    for (a,b),p in scores.items():row.setdefault(a,[]).append((p,b));col.setdefault(b,[]).append((p,a))
    nearest=np.full(len(coords),14.);density=np.zeros(len(coords))
    for t in np.unique(coords[:,0]):
        ids=np.flatnonzero(coords[:,0]==t);old=ids[mapping[ids]>=0];xyz=coords[ids,1:]*SCALE
        tree=cKDTree(xyz);density[ids]=tree.query_ball_point(xyz,7,return_length=True)-1
        if len(old):nearest[ids]=np.minimum(28,cKDTree(coords[old,1:]*SCALE).query(xyz)[0])
    result=[]
    def stats(a):
        a=np.asarray(a,float)
        return [float(a.min()),float(np.median(a)),float(a.mean()),float(a.max()),float(a.std())] if len(a) else [0.]*5
    for path in paths:
        j=0
        while j<len(path):
            if mapping[path[j]]>=0:j+=1;continue
            begin=j
            while j<len(path) and mapping[path[j]]<0:j+=1
            fresh=path[begin:j];left=path[begin-1] if begin else None;right=path[j] if j<len(path) else None
            whole=([left] if left is not None else [])+fresh+([right] if right is not None else [])
            if len(whole)<3:continue
            links=list(zip(whole[:-1],whole[1:]));ps=[scores[a,b] for a,b in links]
            margins=[min(scores[a,b]-max((p for p,k in row[a] if k!=b),default=0),scores[a,b]-max((p for p,k in col[b] if k!=a),default=0)) for a,b in links]
            xyz=coords[whole,1:]*SCALE;speed=np.linalg.norm(np.diff(xyz,axis=0),axis=1);accel=np.linalg.norm(np.diff(xyz,n=2,axis=0),axis=1)
            anchors=int(left is not None)+int(right is not None)
            x=[np.log1p(len(fresh)),anchors,int(left is not None),int(right is not None)]
            for a in (confidence[fresh],ps,margins,speed,accel,nearest[fresh],density[fresh]):x+=stats(a)
            result.append(dict(whole=list(map(int,whole)),fresh=list(map(int,fresh)),anchors=anchors,x=list(map(float,x))))
    return result

def integrate(nodes,edges,coords,mapping,items,scores,threshold,cap_fraction=.01):
    result={k:dict(v) for k,v in nodes.items()};links=set(map(tuple,edges));incoming={b for a,b in links};outgoing={a for a,b in links}
    cap=max(1,int(cap_fraction*len(nodes))) if cap_fraction is not None else len(coords);added=0;accepted=[];next_id=max(nodes,default=-1)+1;created={}
    for index in sorted(range(len(items)),key=lambda i:(-scores[i],items[i]['whole'][0])):
        p=items[index]
        if scores[index]<threshold or added+len(p['fresh'])>cap:continue
        first,last=p['whole'][0],p['whole'][-1]
        if (mapping[first]>=0 and mapping[first] in outgoing) or (mapping[last]>=0 and mapping[last] in incoming):continue
        if any(i in created for i in p['fresh']):continue
        chain=[]
        for i in p['whole']:
            if mapping[i]>=0:
                k=int(mapping[i]);assert np.array_equal(coords[i],[nodes[k][a] for a in ('t','z','y','x')]);chain.append(k)
            else:
                result[next_id]=dict(zip(('t','z','y','x'),map(int,coords[i])));created[i]=next_id;chain.append(next_id);next_id+=1
        new=list(zip(chain[:-1],chain[1:]));assert all(result[b]['t']==result[a]['t']+1 for a,b in new)
        assert not any(a in outgoing or b in incoming for a,b in new)
        links.update(new);outgoing.update(a for a,b in new);incoming.update(b for a,b in new);added+=len(p['fresh']);accepted.append(dict(index=index,edges=new))
    return result,sorted(links),dict(accepted=accepted,added_nodes=added,added_edges=len(links)-len(edges),node_budget=cap,proposals=len(items))

def design(x,mean,scale):
    z=np.clip((np.asarray(x,float)-mean)/scale,-5,5)
    # Pairwise interactions model combinations such as confidence x acceleration.
    i,j=np.triu_indices(z.shape[1]);return np.c_[np.ones(len(z)),z,z[:,i]*z[:,j]]

def fit(x,y,weights):
    x=np.asarray(x,float);y=np.asarray(y,float);weights=np.asarray(weights,float);weights/=weights.sum()
    assert len(x)>=20 and set(y)=={0.,1.} and np.isfinite(x).all()
    mean=np.average(x,axis=0,weights=weights);scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.1)
    a=design(x,mean,scale);ridge=.001;initial=np.zeros(a.shape[1]);initial[0]=np.log(np.average(y,weights=weights)/(1-np.average(y,weights=weights)))
    def objective(w):
        logits=a@w;penalty=w.copy();penalty[0]=0
        loss=np.sum(weights*(np.logaddexp(0,logits)-y*logits))+.5*ridge*np.dot(penalty,penalty)
        grad=a.T@(weights*(expit(logits)-y))+ridge*penalty
        return loss,grad
    r=minimize(objective,initial,method='L-BFGS-B',jac=True,options=dict(maxiter=500,ftol=1e-10,gtol=1e-6))
    if not r.success:raise RuntimeError('Selector fit did not converge: '+str(r.message))
    names=FEATURE_NAMES if x.shape[1]==len(FEATURE_NAMES) else ['feature_'+str(i) for i in range(x.shape[1])]
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=r.x.tolist(),feature_names=names,
        basis='intercept; standardized clipped [-5,5] features; upper triangle including diagonal of pairwise products',
        objective=float(r.fun),iterations=int(r.nit),ridge=ridge,converged=True)

def predict(model,x):
    if not len(x):return np.empty(0)
    return expit(design(x,np.asarray(model['mean']),np.asarray(model['scale']))@np.asarray(model['coef']))
