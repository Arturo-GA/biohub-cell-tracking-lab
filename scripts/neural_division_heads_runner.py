import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from association_cpu_runner import locate
from biohub_lab.neural_division import event_features,development_threshold

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/neural_division_heads');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e037_neural_division.json').read_text());p=locate(inputs,'neural_division_data/result.json');fp=locate(inputs,'neural_division_features/result.json');data=json.loads(p.read_text());features=json.loads(fp.read_text())
    assert data['status']==features['status']=='complete' and data['config']==features['config']==config and features['input_manifest_sha256']==sha(p)
    rows=[]
    for item in data['videos']:
        ep=p.parent/item['events'];assert sha(ep)==item['events_sha256']
        fr=next(r for r in features['outputs'] if r['video']==item['video']);q=fp.parent/fr['file'];assert sha(q)==fr['sha256'] and fr['crops_sha256']==item['crops_sha256']
        with np.load(ep) as d:row=dict(item,**{k:d[k] for k in d.files})
        with np.load(q) as h:row['h']={k:h[k] for k in h.files}
        rows.append(row)
    results=[];choices={}
    for fold in ['44b6','6bba']:
        for arm in ['geometry','initial','sequence']:
            rng=np.random.default_rng(370920)
            def x(row,shuffled=False):
                if arm=='geometry':return row['geometry']
                h=row['h'][fold]
                if shuffled:h=h[rng.permutation(len(h))]
                return np.concatenate([row['geometry'],event_features(h,row['contexts'],arm=='sequence')],axis=1)
            fit=[r for r in rows if r['split']=='fit'];dev=[r for r in rows if r['split']=='development']
            X=np.concatenate([x(r) for r in fit]);y=np.concatenate([r['y'] for r in fit]);D=np.concatenate([x(r) for r in dev]);dy=np.concatenate([r['y'] for r in dev])
            net=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.8,class_weight='balanced',random_state=350920,n_jobs=2);net.fit(X,y)
            score=net.predict_proba(D)[:,1];ap=float(average_precision_score(dy,score));threshold=development_threshold(dy,score)
            shuffle_score=net.predict_proba(np.concatenate([x(r,True) for r in dev]))[:,1];shuffle_ap=float(average_precision_score(dy,shuffle_score))
            gate=arm!='geometry' and threshold is not None and ap>=config['old_static_ap']+.05 and ap>=shuffle_ap+.05
            path=root/(fold+'_'+arm+'_model.pkl')
            with path.open('wb') as out:pickle.dump(net,out)
            record=dict(heldout_embryo=fold,arm=arm,development_ap=ap,shuffled_ap=shuffle_ap,threshold=threshold,gate_passed=gate,model=path.name,model_sha256=sha(path),fit_positive=int(y.sum()),development_positive=int(dy.sum()),feature_dimensions=X.shape[1],development_accepted_tp=int(((dy==1)&(score>=threshold)).sum()) if threshold is not None else 0,development_accepted_fp=int(((dy==0)&(score>=threshold)).sum()) if threshold is not None else 0)
            results.append(record);print('NEURAL_HEAD',json.dumps(record),flush=True)
            if arm=='geometry':assert abs(ap-config['old_geometry_ap'])<1e-12
            np.savez_compressed(root/(fold+'_'+arm+'_development.npz'),y=dy,score=score,shuffled_score=shuffle_score)
        passed=[r for r in results if r['heldout_embryo']==fold and r['gate_passed']]
        choices[fold]=max(passed,key=lambda r:(r['development_ap'],r['arm']=='sequence'))['arm'] if passed else None
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,heads=results,choices=choices,proceed_to_graph=any(v is not None for v in choices.values()),seconds=time.monotonic()-start,feature_manifest_sha256=sha(fp),calibration_labels_read=False,leaderboard_submitted=False),indent=2,allow_nan=False))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
