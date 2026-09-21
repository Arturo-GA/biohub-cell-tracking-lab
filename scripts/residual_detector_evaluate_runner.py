"""CPU: frozen complementary trajectories, full official metrics on 24 movies."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from residual_detector_io import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.residual_tracklets import augment
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_official.metrics import summarise

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e061_protocol.json').read_text())
    pins=json.loads((package/'baseline/e061_evaluate_pins.json').read_text());out=Path('/kaggle/working/residual_detector_evaluation');out.mkdir(exist_ok=True)
    p=one('residual_detector_predictions/result.json');assert sha(p)==pins['predictions_sha256'];pred=json.loads(p.read_text())
    gp=one('temporal_graph_inputs/result.json');assert sha(gp)==pins['graphs_sha256'];old=json.loads(gp.read_text())
    vp=one('visual_validation/result.json').parent;assert sha(vp/'visual.csv')==json.loads((vp/'frozen_predictions.json').read_text())['csv_sha256']['visual']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['evaluation'],out/'data');shapes=shapes_for(data);old16=[r['video'] for r in old['videos']];old8=[v for v in cfg['evaluation'] if v not in old16]
    graphs=read_and_validate(vp/'visual.csv',{v:shapes[v] for v in old8})
    for r in old['videos']:
        path=gp.parent/r['graph'];assert sha(path)==r['graph_sha256']
        with np.load(path) as z:
            nodes={int(k):dict(zip(('t','z','y','x'),map(int,c))) for k,c in zip(z['ids'],z['coords'])};edges=[tuple(map(int,e)) for e in z['edges']]
        graphs[r['video']]=(nodes,edges)
    assert set(graphs)==set(cfg['evaluation'])
    arms=['control','frozen_anchor','frozen_new','adapted_anchor','adapted_new','repeated_new']
    handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};indices={a:0 for a in arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for video in cfg['evaluation']:
            nodes,edges=graphs[video];cached={}
            for model in ('frozen','adapted','repeated'):
                r=next(r for r in pred['records'] if r['video']==video and r['arm']==model);path=p.parent/r['file'];assert sha(path)==r['sha256']
                with np.load(path) as z:cached[model]={k:z[k] for k in ('coords','prob','features')}
            for arm in arms:
                if arm=='control':ns,es,report=nodes,edges,{}
                else:
                    model,mode=arm.split('_');d=cached[model];ns,es,report=augment(nodes,edges,d['coords'],d['prob'],d['features'],standalone=mode=='new')
                for k in sorted(ns):writers[arm].writerow([indices[arm],video,'node',k,*[ns[k][a] for a in ('t','z','y','x')],-1,-1]);indices[arm]+=1
                for a,b in es:writers[arm].writerow([indices[arm],video,'edge',-1,-1,-1,-1,-1,a,b]);indices[arm]+=1
                reports.append(dict(video=video,arm=arm,**report))
            print('ENSEMBLE_RESIDUAL_GRAPH',video,{a:next(r['added_nodes'] for r in reversed(reports) if r['video']==video and r['arm']==a) for a in arms[1:]},flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes,indent=2))
    metrics={};cohorts={}
    for arm in arms:
        m=evaluate_csv(out/(arm+'.csv'),data);assert sha(out/(arm+'.csv'))==hashes[arm]
        (out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        cohorts[arm]={label:summarise([s for s in m['samples'] if s['dataset'] in names]) for label,names in [('old16',old16),('visual8',old8),('44b6',[v for v in graphs if v.startswith('44b6')]),('6bba',[v for v in graphs if v.startswith('6bba')])]}
        print('ENSEMBLE_RESIDUAL_METRIC',arm,metrics[arm],cohorts[arm],flush=True)
    assert abs(cohorts['control']['old16']['score']-.9007529595605399)<1e-10
    assert abs(cohorts['control']['visual8']['score']-.9502832669356378)<1e-10
    passed=[]
    for arm in arms[1:]:
        if metrics[arm]['score']<metrics['control']['score']+cfg['promotion']['minimum_mean_gain']:continue
        if metrics[arm]['edge_tp']<=metrics['control']['edge_tp'] or metrics[arm]['division_fp']>metrics['control']['division_fp']:continue
        if any(cohorts[arm][c]['score']<cohorts['control'][c]['score']-1e-10 for c in cohorts[arm]):continue
        passed.append(arm)
    result=dict(status='complete',config=cfg,metrics=metrics,cohorts=cohorts,reports=reports,passed=passed,
        selected=max(passed,key=lambda a:metrics[a]['score']) if passed else 'control',seconds=time.monotonic()-start,
        leaderboard_submitted=False,scope='24 reused research videos disjoint from current detector fit40; not an independent project holdout. Frozen CSVs before labels.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('ENSEMBLE_RESIDUAL_COMPLETE',passed,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
