import hashlib,json,time
from pathlib import Path
import numpy as np
import tracksdata as td
import zarr
from biohub_lab.division_sequence import candidates,descriptors,features,labels,SCALE
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from association_cpu_runner import locate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    started=time.monotonic();root=Path('/kaggle/working/division_sequence_data');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e035_division_sequence.json').read_text());inputs=Path('/kaggle/input')
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    gp=locate(inputs,'temporal_graph_inputs/result.json');assert sha(gp)==config['graph_manifest_sha256'];graphs=json.loads(gp.read_text())
    keys=td.DEFAULT_ATTR_KEYS;records=[];rng=np.random.default_rng(350920)
    for split in ['fit','development','validation']:
        for video in config[split]:
            graph=td.graph.IndexedRXGraph.from_geff(train/(video+'.geff'))[0]
            rows=sorted(graph.node_attrs().iter_rows(named=True),key=lambda r:r[keys.NODE_ID])
            truth_ids=np.array([r[keys.NODE_ID] for r in rows],np.int64);truth_coords=np.array([[r[k] for k in ['t','z','y','x']] for r in rows],np.float32)
            edge_rows=graph.edge_attrs(attr_keys=[keys.EDGE_SOURCE,keys.EDGE_TARGET]).iter_rows(named=True)
            te=np.asarray([[int(r[keys.EDGE_SOURCE]),int(r[keys.EDGE_TARGET])] for r in edge_rows],np.int64).reshape(-1,2)
            image=zarr.open_group(str(train/(video+'.zarr')),mode='r')['0']
            if split=='validation':
                item=next(r for r in graphs['videos'] if r['video']==video);p=gp.parent/item['graph'];assert sha(p)==item['graph_sha256']
                with np.load(p) as d:ids,coords,original=d['ids'],d['coords'],d['edges']
                ix={int(n):i for i,n in enumerate(ids)};edges=np.asarray([[ix[int(a)],ix[int(b)]] for a,b in original],np.int64).reshape(-1,2)
                identity=np.full(len(ids),-1,np.int64)
                for t in np.unique(coords[:,0]):
                    pi=np.flatnonzero(coords[:,0]==t);gi=np.flatnonzero(truth_coords[:,0]==t)
                    if not len(gi):continue
                    distance=cdist(truth_coords[gi,1:]*SCALE,coords[pi,1:]*SCALE);penalty=(len(gi)+1)*8
                    cost=np.c_[np.where(distance<=7,distance,2*penalty),np.full((len(gi),len(gi)),penalty)]
                    rr,cc=linear_sum_assignment(cost)
                    for gt,pred in zip(rr,cc):
                        if pred<len(pi) and distance[gt,pred]<=7:identity[pi[pred]]=truth_ids[gi[gt]]
            else:
                ids=truth_ids.copy();coords=truth_coords.copy();ix={int(n):i for i,n in enumerate(ids)}
                edges=np.asarray([[ix[int(a)],ix[int(b)]] for a,b in te],np.int64).reshape(-1,2)
                coords[:,1:]+=rng.uniform(-1,1,(len(coords),3))/SCALE
                coords[:,1:]=np.clip(coords[:,1:],0,np.array(image.shape[1:])-1);identity=truth_ids.copy()
            triples,context=candidates(coords,edges,predicted=split=='validation');y=labels(triples,identity,te)
            if split!='validation':
                positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
                if split=='fit' and len(negative)>2000:negative=np.sort(rng.choice(negative,2000,replace=False))
                keep=np.sort(np.r_[positive,negative]);triples,context,y=triples[keep],context[keep],y[keep]
            h=np.zeros((len(coords),6),np.float32);needed=np.unique(context)
            for t in np.unique(coords[needed,0]).astype(int):
                ni=needed[coords[needed,0]==t]
                volume=np.asarray(image[t,:,::4,::4],np.float32)
                h[ni]=descriptors(volume,coords[ni,1:])
            geo,static,sequence=features(coords,h,context)
            path=root/(video+'.npz');np.savez_compressed(path,ids=ids,coords=coords,edges=edges,triples=triples,contexts=context,y=y,geometry=geo,static=static,sequence=sequence)
            counts={int(a):0 for a in truth_ids}
            for a,b in te:counts[int(a)]+=1
            r=dict(video=video,split=split,file=path.name,sha256=sha(path),candidates=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),unknown=int((y<0).sum()),annotated_divisions=sum(c==2 for c in counts.values()),image_nodes=len(needed))
            records.append(r);print('DIVISION_DATA',json.dumps(r),flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,videos=records,seconds=time.monotonic()-started),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
