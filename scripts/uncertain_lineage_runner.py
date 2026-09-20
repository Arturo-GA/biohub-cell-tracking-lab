"""E043 CPU full-graph metric for multi-hypothesis lineage association."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from biohub_lab.uncertain_lineage import reassign_uncertain
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import locate,evaluation_dir
from spatial_probe_features_runner import sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/uncertain_lineage');out.mkdir(exist_ok=True);inputs=Path('/kaggle/input');p=locate(inputs,'temporal_graph_inputs/result.json');f=locate(inputs,'temporal_graph_features/result.json');data=json.loads(p.read_text());features=json.loads(f.read_text())
    assert sha(p)=='c34d3d8daa9a50292001153ba0b8f500ca71b81f756a85cac9e02e1b5826df22';assert sha(f)=='bac1ff43079fb1cb9a6f89d231e00674a0e517056913b0b7bf4254d4aa8deaf6'
    names=[v['video'] for v in data['videos']];train=next(q/'train' for q in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (q/'train').exists());evaluation=evaluation_dir(train,names,out/'data');reports=[]
    handles={a:(out/(a+'.csv')).open('w',newline='') for a in ['control','map','consensus']};writers={a:csv.writer(h) for a,h in handles.items()};counter={a:0 for a in handles}
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for item in data['videos']:
            gp=p.parent/item['graph'];assert sha(gp)==item['graph_sha256']
            with np.load(gp) as g:ids=g['ids'];coords=g['coords'];edges=g['edges']
            fr=next(r for r in features['outputs'] if r['video']==item['video']);fp=f.parent/fr['file'];assert sha(fp)==fr['sha256'] and fr['graph_sha256']==item['graph_sha256'];h=np.load(fp);assert h.shape==(len(ids),64) and np.isfinite(h).all()
            arms,report=reassign_uncertain(ids,coords,edges,h);arms['control']=edges;reports.append(dict(video=item['video'],**report));print('UNCERT_GRAPH',item['video'],report['changed_edges'],flush=True)
            for arm,e in arms.items():
                w=writers[arm]
                for n,c in zip(ids,coords):w.writerow([counter[arm],item['video'],'node',int(n),*map(int,c),-1,-1]);counter[arm]+=1
                for a,b in e:w.writerow([counter[arm],item['video'],'edge',-1,-1,-1,-1,-1,int(a),int(b)]);counter[arm]+=1
    finally:
        for h in handles.values():h.close()
    metrics={}
    for arm in handles:
        metric=evaluate_csv(out/(arm+'.csv'),evaluation);(out/(arm+'_metrics.json')).write_text(json.dumps(metric,indent=2));metrics[arm]=metric['summary'];print('UNCERT_METRIC',arm,metrics[arm],flush=True)
    c=metrics['control'];assert abs(c['score']-.9007529595605399)<1e-10
    passed={a:metrics[a]['score']>=c['score']+.002 and metrics[a]['division_tp']>c['division_tp'] and metrics[a]['division_fp']<=c['division_fp'] for a in ['map','consensus']}
    (out/'result.json').write_text(json.dumps(dict(status='complete',metrics=metrics,passed=passed,reports=reports,seconds=time.monotonic()-start,scope='Reused conditional calibration; fixed node set and incoming-edge coverage, variable division count; perturb-and-MAP frequency is not calibrated Bayesian probability',leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
