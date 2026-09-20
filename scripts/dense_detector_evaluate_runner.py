"""E031 CPU: detector coverage plus complete geometric tracking controls."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from biohub_lab.nucverse_instances import matched_truth
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from association_cpu_runner import locate,evaluation_dir
SCALE=np.array([1.625,.40625,.40625])

def link(coords,radius=14.,minimum_length=6):
    """One-to-one continuation with explicit unmatched alternatives; no GT input."""
    edges=[]
    for t in np.unique(coords[:,0]):
        a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero(coords[:,0]==t+1)
        if not len(a) or not len(b):continue
        x=coords[a,1:]*SCALE;y=coords[b,1:]*SCALE
        neighborhoods=cKDTree(y).query_ball_point(x,radius)
        rows=[];cols=[];cost=[]
        for i,ns in enumerate(neighborhoods):
            for j in ns:rows.append(i);cols.append(j);cost.append(1+float(np.linalg.norm(x[i]-y[j])))
            rows.append(i);cols.append(len(b)+i);cost.append(radius+2)
        mat=coo_matrix((cost,(rows,cols)),shape=(len(a),len(b)+len(a))).tocsr()
        r,c=min_weight_full_bipartite_matching(mat)
        edges.extend((int(a[i]),int(b[j])) for i,j in zip(r,c) if j<len(b))
    parent=np.arange(len(coords));size=np.ones(len(coords),int)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for a,b in edges:
        x,y=find(a),find(b)
        if x!=y:parent[y]=x;size[x]+=size[y]
    keep=np.array([size[find(i)]>=minimum_length for i in range(len(coords))])
    return np.flatnonzero(keep),[(a,b) for a,b in edges if keep[a] and keep[b]]

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/dense_detector_evaluation');root.mkdir(exist_ok=True)
    inputs=Path('/kaggle/input');config=json.loads((package/'baseline/e031_dense_detector.json').read_text())
    protocol=json.loads((package/'baseline/e031_evaluation_protocol.json').read_text())
    source=locate(inputs,'dense_detector_predictions/result.json');inference=json.loads(source.read_text());assert inference['status']=='complete'
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    data=evaluation_dir(train,config['videos'],root/'data');shapes=shapes_for(data)
    control=locate(inputs,'calibration_compare/harmonic_validated.csv')
    assert hashlib.file_digest(control.open('rb'),'sha256').hexdigest()==protocol['control_sha256']
    h=read_and_validate(control,shapes)
    prep=locate(inputs,'dense_detector_inputs/result.json').parent
    report=[];metric={}
    for mode in config['modes']:
        path=root/(mode+'.csv');index=0
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(COLUMNS)
            for video in config['videos']:
                record=next(r for r in inference['files'] if r['video']==video and r['mode']==mode)
                p=source.parent/record['file'];assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==record['sha256']
                with np.load(p,allow_pickle=False) as d:coords=d['coords'];prob=d['probability']
                truth=json.loads((prep/(video+'_truth.json')).read_text());gt=np.asarray(truth['nodes'],float)
                stats=dict(video=video,mode=mode,truth=len(gt),dense_nodes=len(coords),harmonic_nodes=len(h[video][0]),dense_matched=0,harmonic_matched=0,dense_only=0,harmonic_only=0,equal_count_dense_matched=0)
                for t in np.unique(gt[:,1]):
                    g=gt[gt[:,1]==t,2:];chosen=np.flatnonzero(coords[:,0]==t);pred=coords[chosen,1:]
                    harmonic=np.asarray([[n[a] for a in ('z','y','x')] for n in h[video][0].values() if n['t']==t]).reshape(-1,3)
                    dm=matched_truth(pred,g);hm=matched_truth(harmonic,g)
                    order=np.argsort(-prob[chosen],kind='stable')[:len(harmonic)]
                    stats['dense_matched']+=len(dm);stats['harmonic_matched']+=len(hm);stats['dense_only']+=len(dm-hm);stats['harmonic_only']+=len(hm-dm)
                    stats['equal_count_dense_matched']+=len(matched_truth(pred[order],g))
                keep,edges=link(coords,protocol['radius_um'],protocol['minimum_track_length'])
                for n in keep:writer.writerow([index,video,'node',int(n),*map(int,coords[n]),-1,-1]);index+=1
                for a,b in edges:writer.writerow([index,video,'edge',-1,-1,-1,-1,-1,a,b]);index+=1
                stats.update(kept_nodes=len(keep),edges=len(edges));report.append(stats);print('DENSE_CPU',json.dumps(stats),flush=True)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();metric[mode]=evaluate_csv(path,data)
        assert digest==hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        (root/(mode+'_metrics.json')).write_text(json.dumps(metric[mode],indent=2))
    control_metric=evaluate_csv(control,data)
    result=dict(status='complete',seconds=time.monotonic()-start,coverage=report,metrics={m:v['summary'] for m,v in metric.items()},control=control_metric['summary'],scope=config['scope'],tracking_control='Independent detector plus radius14 one-to-one assignment and minimum6 frames; no divisions; not an identical-decoder detector ablation',sparse_unmatched_are_not_false_positives=True,leaderboard_submitted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print('DENSE_EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
