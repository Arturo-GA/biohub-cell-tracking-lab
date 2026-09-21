"""CPU replay on E023's eight reused evaluation videos; predictions before GT."""
import csv,json,sys,time
from pathlib import Path
import numpy as np,zarr
from ensemble_evaluate_runner import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_lab.visual_assignment import assign
from biohub_lab.adaptive_association import descriptors,fuse

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/adaptive_association');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e051_protocol.json').read_text());p=one('visual_validation/result.json');old=json.loads(p.read_text());assert old['status']=='complete'
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    names=json.loads((package/'baseline/e023_validation.json').read_text())['videos'];data=evaluation_dir(train,names,out/'data');shapes=shapes_for(data)
    reference=read_and_validate(p.parent/'harmonic.csv',shapes);frozen=json.loads((p.parent/'frozen_predictions.json').read_text());assert sha(p.parent/'harmonic.csv')==frozen['csv_sha256']['harmonic']
    arms=['control','visual','mutual','adaptive','adaptive_mutual'];files={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in files.items()};counter={a:0 for a in arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for name in names:
            nodes,original=reference[name]
            with np.load(p.parent/(name+'_visual.npz')) as z:visual={tuple(map(int,e)):float(v) for e,v in zip(z['edges'],z['prob'])}
            appearance=descriptors(nodes,zarr.open_group(str(train/(name+'.zarr')),mode='r')['0'])
            for arm in arms:
                if arm=='control':edges,report=original,{}
                elif arm=='visual':edges,report=assign(nodes,original,visual)
                else:edges,report=fuse(nodes,original,visual,appearance,arm)
                for k in sorted(nodes):writers[arm].writerow([counter[arm],name,'node',k,*[nodes[k][axis] for axis in ['t','z','y','x']],-1,-1]);counter[arm]+=1
                for a,b in edges:writers[arm].writerow([counter[arm],name,'edge',-1,-1,-1,-1,-1,a,b]);counter[arm]+=1
                reports.append(dict(video=name,arm=arm,**report))
            print('ENSEMBLE_ADAPTIVE',name,flush=True)
    finally:
        for h in files.values():h.close()
    metrics={}
    for arm in arms:
        read_and_validate(out/(arm+'.csv'),shapes);m=evaluate_csv(out/(arm+'.csv'),data)
        (out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ['edge_tp','edge_fp','edge_fn']})
        print('ENSEMBLE_ADAPTIVE_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-.9436828898822122)<1e-10
    assert abs(metrics['visual']['score']-.9502832669356378)<1e-10
    selected=max(arms[2:],key=lambda a:metrics[a]['score']);improved=metrics[selected]['score']>metrics['visual']['score']+1e-6
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,reports=reports,selected=selected,improved_over_visual=improved,seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Reused E023 eight videos, exploratory, no independent holdout'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
