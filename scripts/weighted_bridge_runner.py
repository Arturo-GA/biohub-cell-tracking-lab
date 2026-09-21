"""E049: frozen weighted complements, CPU full-graph comparison."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from ensemble_evaluate_runner import one,sha
from biohub_lab.weighted_bridge import combine
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import evaluation_dir

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/weighted_bridge');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e049_protocol.json').read_text())
    gp=one('temporal_graph_inputs/result.json');dp=one('consensus_graph/result.json')
    assert sha(gp)==cfg['graphs_sha256'] and sha(dp)==cfg['donors_sha256']
    graphs=json.loads(gp.read_text());donors=json.loads(dp.read_text())
    arms=['control','dense','weighted','motion','weighted_motion'];reports=[]
    handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};counts={a:0 for a in arms}
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for g in graphs['videos']:
            d=next(d for d in donors['videos'] if d['video']==g['video'])
            assert sha(gp.parent/g['graph'])==g['graph_sha256'] and sha(dp.parent/d['file'])==d['sha256']
            with np.load(gp.parent/g['graph']) as z:ids,coords,edges=z['ids'],z['coords'],z['edges']
            with np.load(dp.parent/d['file']) as z:dense,consensus=z['dense'],z['consensus']
            for arm in arms:
                ns,cs,es,report=(ids,coords,edges,{}) if arm=='control' else combine(ids,coords,edges,dense,consensus,arm)
                for n,c in zip(ns,cs):writers[arm].writerow([counts[arm],g['video'],'node',int(n),*map(int,c),-1,-1]);counts[arm]+=1
                for a,b in es:writers[arm].writerow([counts[arm],g['video'],'edge',-1,-1,-1,-1,-1,int(a),int(b)]);counts[arm]+=1
                reports.append(dict(video=g['video'],arm=arm,**report))
            print('ENSEMBLE_WEIGHTED',g['video'],flush=True)
    finally:
        for h in handles.values():h.close()
    inputs=Path('/kaggle/input');train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    evaluation=evaluation_dir(train,[g['video'] for g in graphs['videos']],out/'data');metrics={}
    for arm in arms:
        m=evaluate_csv(out/(arm+'.csv'),evaluation);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ['edge_tp','edge_fp','edge_fn']})
        print('ENSEMBLE_WEIGHTED_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-.9007529595605399)<1e-10
    assert abs(metrics['dense']['score']-.9014086181748129)<1e-10
    selected=max(arms[1:],key=lambda a:(metrics[a]['score'],-metrics[a]['edge_fp']))
    result=dict(status='complete',metrics=metrics,selected=selected,reports=reports,seconds=time.monotonic()-start,config=cfg,leaderboard_submitted=False,scope='Exploratory reused calibration. User explicitly requests a submission of best candidate, including small gains.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
