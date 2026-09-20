import csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from biohub_lab.division_sequence import decode,links,context,SCALE
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import locate,evaluation_dir

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/division_sequence_evaluation');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e035_division_sequence.json').read_text());p=locate(inputs,'division_sequence_data/result.json');manifest=json.loads(p.read_text());assert manifest['status']=='complete' and manifest['config']==config
    rows=[]
    for item in manifest['videos']:
        q=p.parent/item['file'];assert sha(q)==item['sha256']
        with np.load(q) as d:rows.append(dict(item,arrays={k:d[k] for k in d.files}))
    train=next(q/'train' for q in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (q/'train').exists());evaluation=evaluation_dir(train,config['validation'],root/'data')
    results={};metrics={};edits={};coverage=[];division_audit=[]
    for r in rows:
        if r['split']!='validation':continue
        d=r['arrays'];existing=set(map(tuple,d['edges']));positive=d['triples'][d['y']==1]
        present=sum((int(m),int(a)) in existing and (int(m),int(b)) in existing for m,a,b in positive)
        coverage.append(dict(video=r['video'],annotated_divisions=r['annotated_divisions'],represented_positive=len(positive),already_connected=int(present),potential_missing=int(len(positive)-present)))
        # Ground-truth audit only: never feeds candidate generation, fitting or decoding.
        import tracksdata as td
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        truth=td.graph.IndexedRXGraph.from_geff(train/(r['video']+'.geff'))[0];keys=td.DEFAULT_ATTR_KEYS
        nr=list(truth.node_attrs().iter_rows(named=True));tc=np.array([[v[k] for k in ['t','z','y','x']] for v in nr]);ti=np.array([v[keys.NODE_ID] for v in nr]);mapping={}
        for t in np.unique(tc[:,0]):
            a=np.flatnonzero(tc[:,0]==t);b=np.flatnonzero(d['coords'][:,0]==t)
            if not len(b):continue
            distances=cdist(tc[a,1:]*SCALE,d['coords'][b,1:]*SCALE);penalty=(len(a)+1)*8
            cost=np.c_[np.where(distances<=7,distances,2*penalty),np.full((len(a),len(a)),penalty)]
            rr,cc=linear_sum_assignment(cost)
            for i,j in zip(rr,cc):
                if j<len(b) and distances[i,j]<=7:mapping[int(ti[a[i]])]=int(b[j])
        children={}
        for e in truth.edge_attrs(attr_keys=[keys.EDGE_SOURCE,keys.EDGE_TARGET]).iter_rows(named=True):children.setdefault(int(e[keys.EDGE_SOURCE]),[]).append(int(e[keys.EDGE_TARGET]))
        previous,following=links(d['edges']);proposed={tuple([int(m),*sorted([int(a),int(b)])]) for m,a,b in d['triples']}
        for mother,daughters in children.items():
            if len(daughters)!=2:continue
            event=dict(video=r['video'],truth_mother=mother,missing_node_ids=[n for n in [mother,*daughters] if n not in mapping])
            if not event['missing_node_ids']:
                m=mapping[mother];a,b=sorted(mapping[n] for n in daughters)
                event.update(represented=(m,a,b) in proposed,complete_context=context((m,a,b),d['coords'],previous,following) is not None,daughter_has_other_parent=any(previous[c] and previous[c]!=[m] for c in [a,b]),mother_has_other_child=not set(following[m]).issubset({a,b}),already_connected=(m,a) in existing and (m,b) in existing)
            division_audit.append(event)
    print('DIVISION_HEAD candidate_coverage',coverage,flush=True)
    def x(d,arm):
        return d['geometry'] if arm=='geometry' else np.concatenate([d['geometry'],d['static' if arm=='static' else 'sequence']],axis=1)
    for arm in ['control','geometry','static','sequence']:
        threshold=float('inf');net=None
        if arm!='control':
            fit=[r['arrays'] for r in rows if r['split']=='fit'];dev=[r['arrays'] for r in rows if r['split']=='development']
            X=np.concatenate([x(d,arm) for d in fit]);Y=np.concatenate([d['y'] for d in fit]);assert set(Y)=={0,1}
            net=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.8,class_weight='balanced',random_state=350920,n_jobs=2);net.fit(X,Y)
            dx=np.concatenate([x(d,arm) for d in dev]);dy=np.concatenate([d['y'] for d in dev]);scores=net.predict_proba(dx)[:,1]
            options=[]
            for value in np.unique(scores):
                select=scores>=value;tp=int(((dy==1)&select).sum());fp=int(((dy==0)&select).sum())
                if tp>=2 and fp==0:options.append((tp,float(value)))
            if options:threshold=sorted(options,key=lambda v:(-v[0],-v[1]))[0][1]
            results[arm]=dict(fit_positive=int(Y.sum()),fit_negative=int((Y==0).sum()),development_positive=int(dy.sum()),development_ap=float(average_precision_score(dy,scores)) if dy.any() else None,threshold=threshold if np.isfinite(threshold) else None,gate_passed=bool(options),development_accepted_tp=int(((dy==1)&(scores>=threshold)).sum()),development_accepted_fp=int(((dy==0)&(scores>=threshold)).sum()))
            with (root/(arm+'_model.pkl')).open('wb') as out:pickle.dump(net,out)
            print('DIVISION_HEAD',arm,results[arm],flush=True)
        path=root/(arm+'.csv');counter=0;edits[arm]=[];cy=[];cs=[]
        with path.open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(COLUMNS)
            for r in rows:
                if r['split']!='validation':continue
                d=r['arrays'];edges=d['edges']
                if arm!='control' and len(d['triples']):
                    score=net.predict_proba(x(d,arm))[:,1];known=d['y']>=0;cy.extend(d['y'][known].tolist());cs.extend(score[known].tolist())
                    edges,changed=decode(edges,d['triples'],score,threshold,d['coords']);edits[arm].append(dict(video=r['video'],events=changed))
                for n,c in zip(d['ids'],d['coords']):writer.writerow([counter,r['video'],'node',int(n),*map(int,c),-1,-1]);counter+=1
                for a,b in edges:writer.writerow([counter,r['video'],'edge',-1,-1,-1,-1,-1,int(d['ids'][a]),int(d['ids'][b])]);counter+=1
        if arm!='control':results[arm].update(calibration_candidate_positive=sum(cy),calibration_known_candidates=len(cy),calibration_ap=float(average_precision_score(cy,cs)) if sum(cy) else None)
        metrics[arm]=evaluate_csv(path,evaluation);(root/(arm+'_metrics.json')).write_text(json.dumps(metrics[arm],indent=2));print('DIVISION_METRIC',arm,metrics[arm]['summary'],flush=True)
    assert abs(metrics['control']['summary']['score']-config['control_score'])<1e-10
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,data_manifest_sha256=sha(p),candidate_coverage=coverage,division_audit=division_audit,heads=results,edits=edits,metrics={a:m['summary'] for a,m in metrics.items()},seconds=time.monotonic()-start,leaderboard_submitted=False),indent=2,allow_nan=False))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
