import json,sys,time
from pathlib import Path
import zarr
from residual_detector_io import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_lab.appearance_swap import describe,refine
from trajectory_selector_select_runner import write_graphs,counts
from biohub_official.metrics import summarise

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e065_protocol.json').read_text());out=Path('/kaggle/working/appearance_swap_evaluation');out.mkdir(exist_ok=True)
    p=one('residual_detector_evaluation/result.json');assert sha(p)==cfg['control_manifest_sha256'];previous=json.loads(p.read_text());assert sha(p.parent/'control.csv')==json.loads((p.parent/'frozen_predictions.json').read_text())['control']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data);base=read_and_validate(p.parent/'control.csv',shapes);graphs={'control':base,**{a:{} for a in cfg['arms']}};reports=[]
    for video,(nodes,edges) in base.items():
        image=zarr.open_group(str(train/(video+'.zarr')),mode='r')['0'];features=describe(nodes,image)
        for arm,options in cfg['arms'].items():
            es,report=refine(nodes,edges,features,**options);graphs[arm][video]=(nodes,es);reports.append(dict(video=video,arm=arm,**report))
        print('ENSEMBLE_APPEARANCE_CENTERS',video,reports[-3:],flush=True)
    frozen={}
    for arm,g in graphs.items():path=out/(arm+'.csv');write_graphs(path,g);frozen[arm]=sha(path)
    (out/'frozen_predictions.json').write_text(json.dumps(frozen,indent=2));metrics={};cohorts={}
    for arm in graphs:
        m=evaluate_csv(out/(arm+'.csv'),data);metrics[arm]=counts(m);cohorts[arm]={e:summarise([s for s in m['samples'] if s['dataset'].startswith(e)]) for e in ('44b6','6bba')};(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));assert sha(out/(arm+'.csv'))==frozen[arm];print('ENSEMBLE_APPEARANCE_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-previous['metrics']['control']['score'])<1e-10
    changed=[a for a in cfg['arms'] if frozen[a]!=frozen['control']];assert changed,'No new predictions'
    selected=max(changed,key=lambda a:metrics[a]['score'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,cohorts=cohorts,reports=reports,selected=selected,delta=metrics[selected]['score']-metrics['control']['score'],seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Exploratory selection on 24 reused research videos. User explicitly requested a new submission; selected means best NEW candidate, not proof of improvement.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
