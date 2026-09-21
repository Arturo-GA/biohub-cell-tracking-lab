"""CPU integration of new detections supported by pretrained neural associations."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from residual_detector_io import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv
from biohub_lab.neural_complement import neural_chains
from biohub_lab.residual_tracklets import augment
from biohub_official.metrics import summarise

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e062_protocol.json').read_text());pins=json.loads((package/'baseline/e062_evaluate_pins.json').read_text())
    out=Path('/kaggle/working/neural_complement_evaluation');out.mkdir(exist_ok=True)
    pp=one('neural_complement_predictions/result.json');assert sha(pp)==pins['inference_sha256'];pred=json.loads(pp.read_text())
    up=one('neural_complement_inputs/result.json');union=json.loads(up.read_text());gp=one('residual_detector_evaluation/result.json');assert sha(gp)==cfg['evaluation_sha256']
    previous=json.loads(gp.read_text());shapes={r['video']:r['shape'] for r in union['records']}
    path=gp.parent/'control.csv';assert sha(path)==json.loads((gp.parent/'frozen_predictions.json').read_text())['control'];graphs=read_and_validate(path,shapes)
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');arms=['control','neural_anchor','neural_new'];reports=[]
    handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};ids={a:0 for a in arms}
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for video,(nodes,original) in graphs.items():
            u=next(r for r in union['records'] if r['video']==video);r=next(r for r in pred['records'] if r['video']==video)
            assert sha(up.parent/u['file'])==u['sha256']==r['union_sha256'];assert sha(pp.parent/r['file'])==r['sha256']
            with np.load(up.parent/u['file']) as z:coords,mapping,confidence=z['coords'],z['mapping'],z['prob']
            with np.load(pp.parent/r['file']) as z:paths=neural_chains(coords,z['edges'],z['prob'])
            for arm in arms:
                if arm=='control':ns,es,report=nodes,original,{}
                else:ns,es,report=augment(nodes,original,coords,confidence,np.zeros((len(coords),1),np.float32),arm=='neural_new',paths,mapping)
                for k in sorted(ns):writers[arm].writerow([ids[arm],video,'node',k,*[ns[k][a] for a in ('t','z','y','x')],-1,-1]);ids[arm]+=1
                for a,b in es:writers[arm].writerow([ids[arm],video,'edge',-1,-1,-1,-1,-1,a,b]);ids[arm]+=1
                reports.append(dict(video=video,arm=arm,**report))
            print('ENSEMBLE_NEURAL_GRAPH',video,len(paths),reports[-2:],flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes,indent=2))
    old8=json.loads((package/'baseline/e023_validation.json').read_text())['videos'];old16=[v for v in graphs if v not in old8];metrics={};cohorts={}
    for arm in arms:
        m=evaluate_csv(out/(arm+'.csv'),data);assert sha(out/(arm+'.csv'))==hashes[arm]
        (out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        cohorts[arm]={label:summarise([s for s in m['samples'] if s['dataset'] in names]) for label,names in [('old16',old16),('visual8',old8),('44b6',[v for v in graphs if v.startswith('44b6')]),('6bba',[v for v in graphs if v.startswith('6bba')])]}
        print('ENSEMBLE_NEURAL_METRIC',arm,metrics[arm],cohorts[arm],flush=True)
    assert abs(metrics['control']['score']-previous['metrics']['control']['score'])<1e-10
    passed=[a for a in arms[1:] if metrics[a]['score']>=metrics['control']['score']+.001 and metrics[a]['edge_tp']>metrics['control']['edge_tp']
        and metrics[a]['division_fp']<=metrics['control']['division_fp'] and all(cohorts[a][c]['score']>=cohorts['control'][c]['score']-1e-10 for c in cohorts[a])]
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,cohorts=cohorts,reports=reports,passed=passed,
        selected=max(passed,key=lambda a:metrics[a]['score']) if passed else 'control',seconds=time.monotonic()-start,leaderboard_submitted=False,
        scope='Exploratory 24 reused videos; E061 detector fit disjoint. No claim of independent public tracker pretraining.'),indent=2))
    print('ENSEMBLE_NEURAL_COMPLETE',passed,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
