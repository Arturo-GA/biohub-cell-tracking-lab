"""E046 combine complementary donor bridges with E033 reassignment on augmented graphs."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from temporal_bridge_runner import one,sha,checked
from biohub_lab.temporal_bridge import augment
from biohub_lab.temporal_graph import reassign
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import evaluation_dir

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e046_protocol.json').read_text());sources={}
    for key,item in cfg['sources'].items():
        p=one(item['path']);assert sha(p)==item['sha256'];sources[key]=(p,json.loads(p.read_text()))
    gp,graphs=sources['graphs'];pp,prep=sources['donors'];fp,features=sources['donor_features'];hp,hs=sources['graph_features']
    out=Path('/kaggle/working/graph_complements');out.mkdir(exist_ok=True)
    inputs=Path('/kaggle/input');train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    evaluation=evaluation_dir(train,[g['video'] for g in graphs['videos']],out/'data');metrics={};changes={};hashes={}
    for arm in ['control','geometry_bridge','visual_bridge','geometry_bridge_encoder','visual_bridge_encoder']:
        path=out/(arm+'.csv');counter=0;changes[arm]=[]
        with path.open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(COLUMNS)
            for gi in graphs['videos']:
                with np.load(checked(gp,gi,'graph','graph_sha256')) as g:ids,coords,edges=g['ids'],g['coords'],g['edges']
                if arm!='control':
                    item=next(r for r in prep['videos'] if r['video']==gi['video']);dense=np.load(checked(pp,item,'file','sha256'))
                    fr=next(r for r in features['videos'] if r['video']==gi['video']);assert fr['donor_sha256']==item['sha256']
                    hr=next(r for r in hs['outputs'] if r['video']==gi['video']);assert hr['graph_sha256']==gi['graph_sha256']
                    h=np.load(checked(hp,hr,'file','sha256'));dh=np.load(checked(fp,fr,'file','sha256'));n=len(ids)
                    ids,coords,edges,report=augment(ids,coords,edges,dense,item['proposals'],h if arm.startswith('visual') else None,dh if arm.startswith('visual') else None)
                    if arm.endswith('encoder'):
                        # Match each actual added node back to its cached donor embedding.
                        mapping={tuple(c):i for i,c in enumerate(dense)};assert len(mapping)==len(dense)
                        new_h=np.vstack([h,dh[[mapping[tuple(c)] for c in coords[n:]]]]) if len(coords)>n else h
                        edges,rerank=reassign(ids,coords,edges,new_h,True,1.)
                        report.update(rerank)
                    changes[arm].append(dict(video=gi['video'],**report))
                for node,c in zip(ids,coords):writer.writerow([counter,gi['video'],'node',int(node),*map(int,c),-1,-1]);counter+=1
                for a,b in edges:writer.writerow([counter,gi['video'],'edge',-1,-1,-1,-1,-1,int(a),int(b)]);counter+=1
        hashes[arm]=sha(path);metric=evaluate_csv(path,evaluation);assert sha(path)==hashes[arm]
        (out/(arm+'_metrics.json')).write_text(json.dumps(metric,indent=2));metrics[arm]=dict(metric['summary'],**{k:sum(r[k] for r in metric['samples']) for k in ['edge_tp','edge_fp','edge_fn']});print('ENSEMBLE_GRAPH',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-.9007529595605399)<1e-10
    assert abs(metrics['geometry_bridge']['score']-.901013522)<1e-8
    c=metrics['control'];best=max(metrics[a]['score'] for a in ['geometry_bridge','visual_bridge'])
    passed={a:metrics[a]['score']>=best+.001 and metrics[a]['edge_tp']>max(metrics[x]['edge_tp'] for x in ['geometry_bridge','visual_bridge']) and metrics[a]['edge_fp']<=c['edge_fp'] and metrics[a]['division_tp']>=c['division_tp'] and metrics[a]['division_fp']<=c['division_fp'] for a in ['geometry_bridge_encoder','visual_bridge_encoder']}
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,changes=changes,csv_sha256=hashes,passed=passed,seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Reused16 calibration. All candidate graphs produced without annotations; no claim of independent validation.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
