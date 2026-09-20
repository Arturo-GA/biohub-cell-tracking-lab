import csv,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from association_cpu_runner import locate,evaluation_dir
from biohub_lab.neural_division import event_features
from biohub_lab.division_sequence import decode
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/neural_division_graph');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e037_neural_division.json').read_text());pins=json.loads((package/'baseline/e037_graph_pins.json').read_text())
    hp=locate(inputs,'neural_division_heads/result.json');heads=json.loads(hp.read_text());assert heads['status']=='complete' and heads['config']==config and heads['proceed_to_graph']
    p=locate(inputs,'division_sequence_data/result.json');assert sha(p)==config['data_manifest_sha256'];data=json.loads(p.read_text())
    fp=locate(inputs,'temporal_graph_features/result.json');assert sha(fp)==pins['features_manifest_sha256'];feature=json.loads(fp.read_text());models={}
    for fold,arm in heads['choices'].items():
        if arm is None:continue
        record=next(r for r in heads['heads'] if r['heldout_embryo']==fold and r['arm']==arm);assert record['gate_passed'] and record['threshold'] is not None
        q=hp.parent/record['model'];assert sha(q)==record['model_sha256']
        with q.open('rb') as f:models[fold]=(pickle.load(f),record)
    rows=[r for r in data['videos'] if r['split']=='validation'];train=next(q/'train' for q in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (q/'train').exists());evaluation=evaluation_dir(train,[r['video'] for r in rows],root/'data');metrics={};changes=[]
    for arm in ['control','neural']:
        path=root/(arm+'.csv');counter=0
        with path.open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(COLUMNS)
            for r in rows:
                q=p.parent/r['file'];assert sha(q)==r['sha256']
                with np.load(q) as d:ids,coords,edges,contexts,triples,geometry=[d[k] for k in ['ids','coords','edges','contexts','triples','geometry']]
                fold=r['video'].split('_')[0]
                if arm=='neural' and fold in models and len(triples):
                    record=next(v for v in feature['outputs'] if v['video']==r['video']);q=fp.parent/record['file'];assert sha(q)==record['sha256'];h=np.load(q);assert h.shape==(len(ids),64)
                    net,chosen=models[fold];X=np.concatenate([geometry,event_features(h,contexts,chosen['arm']=='sequence')],axis=1)
                    scores=net.predict_proba(X)[:,1];edges,edits=decode(edges,triples,scores,chosen['threshold'],coords);changes.append(dict(video=r['video'],arm=chosen['arm'],threshold=chosen['threshold'],events=edits))
                for n,c in zip(ids,coords):writer.writerow([counter,r['video'],'node',int(n),*map(int,c),-1,-1]);counter+=1
                for a,b in edges:writer.writerow([counter,r['video'],'edge',-1,-1,-1,-1,-1,int(ids[a]),int(ids[b])]);counter+=1
        metrics[arm]=evaluate_csv(path,evaluation);(root/(arm+'_metrics.json')).write_text(json.dumps(metrics[arm],indent=2));print('NEURAL_GRAPH',arm,metrics[arm]['summary'],flush=True)
    assert abs(metrics['control']['summary']['score']-pins['control_score'])<1e-10
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,heads_manifest_sha256=sha(hp),metrics={a:m['summary'] for a,m in metrics.items()},changes=changes,seconds=time.monotonic()-start,leaderboard_submitted=False),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
