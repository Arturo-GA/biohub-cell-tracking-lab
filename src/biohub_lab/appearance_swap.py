"""Label-free native-image tracklet descriptors and degree-preserving 2-opt."""
from collections import Counter
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

SCALE=np.array([1.625,.40625,.40625])
ARMS={
    'strict':dict(appearance_gain=.30,length_slack=.5,motion_slack=0.),
    'balanced':dict(appearance_gain=.20,length_slack=1.5,motion_slack=.5),
    'appearance':dict(appearance_gain=.40,length_slack=3.,motion_slack=1.),
}

def describe(nodes,image):
    if not nodes:return np.empty((0,127),np.float32)
    ids=sorted(nodes);coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids])
    offsets=np.stack(np.meshgrid(*([np.linspace(-2,2,5)]*3),indexing='ij'),-1).reshape(-1,3)/SCALE
    result=np.zeros((len(ids),127),np.float32)
    for t in np.unique(coords[:,0]):
        rows=np.flatnonzero(coords[:,0]==t);frame=np.asarray(image[int(t)],np.float32)
        for start in range(0,len(rows),256):
            at=rows[start:start+256];points=coords[at,1:,None]+offsets.T[None]
            values=map_coordinates(frame,points.transpose(1,0,2).reshape(3,-1),order=1,mode='nearest').reshape(len(at),-1)
            mean=values.mean(1);std=values.std(1);patch=(values-mean[:,None])/np.maximum(std[:,None],1.)
            result[at,:125]=patch/np.sqrt(125)
            result[at,125]=np.log1p(np.maximum(mean,0))*.25
            result[at,126]=np.log1p(std)*.25
        # Remove frame-wide illumination changes without using annotation data.
        result[rows,125:]-=np.median(result[rows,125:],axis=0)
    return result

def refine(nodes,edges,features,appearance_gain,length_slack,motion_slack):
    if not edges:return list(edges),dict(proposed_pairs=0,accepted_swaps=0,changed_edges=0,appearance_gain_sum=0,nodes_changed=False,degree_sequence_changed=False)
    ids=sorted(nodes);ix={k:i for i,k in enumerate(ids)};coords=np.array([[nodes[k][a] for a in ('t','z','y','x')] for k in ids]);xyz=coords[:,1:]*SCALE
    pairs=np.array([(ix[a],ix[b]) for a,b in edges],int).reshape(-1,2);inc=Counter(pairs[:,1]);out=Counter(pairs[:,0])
    parent={int(b):int(a) for a,b in pairs if inc[b]==1 and out[a]==1};child={int(a):int(b) for a,b in pairs if out[a]==1 and inc[b]==1}
    before=features.copy();after=features.copy()
    for b,a in parent.items():before[b]=(features[b]+features[a])*.5
    for a,b in child.items():after[a]=(features[a]+features[b])*.5
    replacements={};proposed=accepted=0;used=set();gains=[]
    eligible=np.array([out[a]==1 and inc[b]==1 and a in parent and b in child and coords[b,0]==coords[a,0]+1 for a,b in pairs])
    for t in np.unique(coords[:,0]):
        valid=pairs[eligible&(coords[pairs[:,0],0]==t)]
        if len(valid)<2:continue
        valid=np.asarray(valid);near=cKDTree(xyz[valid[:,0]]).query_pairs(12.,output_type='ndarray');options=[]
        if not len(near):continue
        a,b=valid[near[:,0]].T;c,d=valid[near[:,1]].T
        norm=lambda x:np.linalg.norm(x,axis=1)
        old=norm(xyz[b]-xyz[a])+norm(xyz[d]-xyz[c]);n1=norm(xyz[d]-xyz[a]);n2=norm(xyz[b]-xyz[c])
        mask=(n1<=14)&(n2<=14)&(n1+n2<=old+length_slack)
        a,b,c,d=[x[mask] for x in (a,b,c,d)]
        app=lambda x,y:np.sum((before[x]-after[y])**2,axis=1)
        current=app(a,b)+app(c,d);alternative=app(a,d)+app(c,b)
        mask=(current-alternative>=.10)&(alternative<=(1-appearance_gain)*current)
        gain=(current-alternative)[mask];a,b,c,d=[x[mask] for x in (a,b,c,d)]
        if not len(a):continue
        def motion(x,y):
            v=xyz[y]-xyz[x];left=xyz[x]-xyz[[parent[int(k)] for k in x]];right=xyz[[child[int(k)] for k in y]]-xyz[y]
            return norm(v-left)+norm(right-v)
        mask=motion(a,d)+motion(c,b)<=motion(a,b)+motion(c,d)+motion_slack
        options=[(float(g),int(aa),int(bb),int(cc),int(dd)) for g,aa,bb,cc,dd in zip(gain[mask],a[mask],b[mask],c[mask],d[mask])];proposed+=len(options)
        for gain,a,b,c,d in sorted(options,reverse=True):
            # Disjoint four-frame neighborhoods avoid mutually inconsistent swaps.
            neighborhood={a,b,c,d,parent[a],parent[c],child[b],child[d]}
            if neighborhood&used:continue
            used.update(neighborhood);replacements[a]=d;replacements[c]=b;accepted+=1;gains.append(gain)
    new=[(ids[int(a)],ids[replacements.get(int(a),int(b))]) for a,b in pairs]
    assert Counter(a for a,b in new)==Counter(a for a,b in edges)
    assert Counter(b for a,b in new)==Counter(b for a,b in edges)
    assert len(set(new))==len(new)
    return new,dict(proposed_pairs=proposed,accepted_swaps=accepted,changed_edges=accepted*2,appearance_gain_sum=sum(gains),nodes_changed=False,degree_sequence_changed=False)
