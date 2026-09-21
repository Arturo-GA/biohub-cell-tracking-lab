"""E055 CPU-only selective ensemble; labels used only after freezing CSVs."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from ensemble_evaluate_runner import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_lab.visual_assignment import assign
from biohub_lab.selective_repair import select

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/selective_repair');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e055_protocol.json').read_text());p=one('visual_validation/result.json');assert json.loads(p.read_text())['status']=='complete'
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data);graphs=read_and_validate(p.parent/'harmonic.csv',shapes)
    frozen=json.loads((p.parent/'frozen_predictions.json').read_text());assert sha(p.parent/'harmonic.csv')==frozen['csv_sha256']['harmonic']
    arms=['control','visual','confidence','trajectory','joint'];handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};ids={a:0 for a in arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for name in cfg['videos']:
            nodes,original=graphs[name]
            with np.load(p.parent/(name+'_visual.npz')) as z:visual={tuple(map(int,e)):float(v) for e,v in zip(z['edges'],z['prob'])}
            candidate,_=assign(nodes,original,visual)
            for arm in arms:
                if arm=='control':edges,report=original,{}
                elif arm=='visual':edges,report=candidate,{}
                else:edges,report=select(nodes,original,candidate,visual,arm)
                for k in sorted(nodes):writers[arm].writerow([ids[arm],name,'node',k,*[nodes[k][a] for a in ('t','z','y','x')],-1,-1]);ids[arm]+=1
                for a,b in edges:writers[arm].writerow([ids[arm],name,'edge',-1,-1,-1,-1,-1,a,b]);ids[arm]+=1
                reports.append(dict(video=name,arm=arm,**report))
            print('ENSEMBLE_SELECTIVE',name,flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes))
    metrics={}
    for arm in arms:
        read_and_validate(out/(arm+'.csv'),shapes);m=evaluate_csv(out/(arm+'.csv'),data);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        print('ENSEMBLE_SELECTIVE_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-.9436828898822122)<1e-10 and abs(metrics['visual']['score']-.9502832669356378)<1e-10
    selected=max(arms,key=lambda a:metrics[a]['score'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,reports=reports,selected=selected,improved_over_visual=metrics[selected]['score']>metrics['visual']['score']+1e-8,seconds=time.monotonic()-start,scope='Eight repeatedly used videos, exploratory; no independent holdout',leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
