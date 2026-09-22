"""Frozen learned selector on the actual Harmonic/E062 union, CPU only."""
import json,sys,time
from pathlib import Path
import numpy as np
from residual_detector_io import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import read_and_validate
from biohub_lab.evaluate import evaluate_csv
from biohub_lab.neural_complement import neural_chains
from biohub_lab.residual_tracklets import augment
from biohub_lab.trajectory_selector import proposals,integrate,predict
from trajectory_selector_select_runner import write_graphs,counts
from biohub_official.metrics import summarise

def conditional_selection(root,cfg):
    """Calibrate each known embryo using development only; unknown -> no addition."""
    cached={a:json.loads((root/(a+'_metrics.json')).read_text())['samples'] for a in ['control']+[f'learned_{t}' for t in cfg['thresholds']]}
    thresholds={};chosen=[];reports={}
    for embryo in sorted({v.split('_')[0] for v in cfg['fit']}):
        samples={a:[s for s in rows if s['dataset'].split('_')[0]==embryo] for a,rows in cached.items()}
        summaries={a:counts(dict(summary=summarise(rows),samples=rows)) for a,rows in samples.items()};b=summaries['control']
        eligible=[a for a,s in summaries.items() if a!='control' and s['score']>b['score']+1e-6 and s['edge_tp']>b['edge_tp'] and s['division_fp']<=b['division_fp']]
        arm=max(eligible,key=lambda a:(summaries[a]['score'],float(a.split('_',1)[1]))) if eligible else 'control'
        thresholds[embryo]=float(arm.split('_',1)[1]) if eligible else 1.01;chosen+=samples[arm];reports[embryo]=dict(selected=arm,control=b,candidate=summaries[arm])
    result=counts(dict(summary=summarise(chosen),samples=chosen));base=counts(dict(summary=summarise(cached['control']),samples=cached['control']))
    return thresholds,dict(metrics=result,control=base,by_embryo=reports,passed=result['score']>base['score']+1e-6 and result['edge_tp']>base['edge_tp'])

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e063_protocol.json').read_text());pins=json.loads((package/'baseline/e063_evaluate_pins.json').read_text())
    out=Path('/kaggle/working/selector_evaluation');out.mkdir(exist_ok=True)
    sp=one('trajectory_selector/result.json');assert sha(sp)==pins['SELECT'];selection=json.loads(sp.read_text());mp=sp.parent/selection['model'];assert sha(mp)==selection['model_sha256'];model=json.loads(mp.read_text())
    threshold=selection['threshold'] if selection['development_passed'] else selection['diagnostic_threshold']
    thresholds,conditional_dev=conditional_selection(sp.parent,cfg)
    (out/'frozen_conditional_development.json').write_text(json.dumps(dict(thresholds=thresholds,development=conditional_dev,unknown_embryo_threshold=1.01),indent=2))
    pp=one('neural_complement_predictions/result.json');up=one('neural_complement_inputs/result.json');gp=one('residual_detector_evaluation/result.json')
    assert sha(pp)==pins['E062_INFER'] and sha(up)==pins['E062_PREPARE'] and sha(gp)==pins['E061_EVALUATE']
    pred=json.loads(pp.read_text());union=json.loads(up.read_text());previous=json.loads(gp.read_text());shapes={r['video']:r['shape'] for r in union['records']}
    assert sha(gp.parent/'control.csv')==json.loads((gp.parent/'frozen_predictions.json').read_text())['control']
    base=read_and_validate(gp.parent/'control.csv',shapes);graphs={'control':base,'rule':{},'learned':{},'conditional':{}};reports=[]
    assert set(base)==set(cfg['evaluation']) and not set(base)&set(cfg['fit']+cfg['development'])
    for video,(nodes,edges) in base.items():
        u=next(r for r in union['records'] if r['video']==video);r=next(r for r in pred['records'] if r['video']==video)
        assert sha(up.parent/u['file'])==u['sha256']==r['union_sha256'] and sha(pp.parent/r['file'])==r['sha256']
        with np.load(up.parent/u['file']) as z:coords,mapping,confidence=z['coords'],z['mapping'],z['prob']
        with np.load(pp.parent/r['file']) as z:candidates,prob=z['edges'],z['prob']
        paths=neural_chains(coords,candidates,prob);items=proposals(coords,mapping,confidence,paths,candidates,prob);scores=predict(model,[p['x'] for p in items])
        ns,es,report=integrate(nodes,edges,coords,mapping,items,scores,threshold,cfg['node_budget_fraction']);graphs['learned'][video]=(ns,es)
        reports.append(dict(video=video,**{k:v for k,v in report.items() if k!='accepted'},accepted=len(report['accepted']),above_threshold=int((scores>=threshold).sum())))
        local_threshold=thresholds.get(video.split('_')[0],1.01)
        ns,es,conditional_report=integrate(nodes,edges,coords,mapping,items,scores,local_threshold,cfg['node_budget_fraction']);graphs['conditional'][video]=(ns,es)
        reports[-1]['conditional']=dict(threshold=local_threshold,accepted=len(conditional_report['accepted']),added_nodes=conditional_report['added_nodes'],added_edges=conditional_report['added_edges'])
        ns,es,_=augment(nodes,edges,coords,confidence,np.zeros((len(coords),1)),True,paths,mapping);graphs['rule'][video]=(ns,es)
        print('ENSEMBLE_SELECTOR_GRAPH',reports[-1],flush=True)
    frozen={}
    for arm,gs in graphs.items():path=out/(arm+'.csv');write_graphs(path,gs);frozen[arm]=sha(path)
    (out/'frozen_predictions.json').write_text(json.dumps(dict(model_sha256=sha(mp),threshold=threshold,conditional_thresholds=thresholds,graphs=frozen),indent=2))
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());data=evaluation_dir(train,cfg['evaluation'],out/'data');metrics={};cohorts={}
    visual=json.loads((package/'baseline/e023_validation.json').read_text())['videos']
    for arm in graphs:
        m=evaluate_csv(out/(arm+'.csv'),data);metrics[arm]=counts(m);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));assert sha(out/(arm+'.csv'))==frozen[arm]
        cohorts[arm]={label:summarise([s for s in m['samples'] if s['dataset'] in names]) for label,names in [('old16',[v for v in base if v not in visual]),('visual8',visual),('44b6',[v for v in base if v.startswith('44b6')]),('6bba',[v for v in base if v.startswith('6bba')])]}
        print('ENSEMBLE_SELECTOR_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['control']['score']-previous['metrics']['control']['score'])<1e-10
    qualified={'learned':selection['development_passed'],'conditional':conditional_dev['passed']}
    passed=[a for a in qualified if qualified[a] and metrics[a]['score']>=metrics['control']['score']+.001 and metrics[a]['edge_tp']>metrics['control']['edge_tp'] and metrics[a]['division_fp']<=metrics['control']['division_fp'] and all(cohorts[a][c]['score']>=cohorts['control'][c]['score'] for c in cohorts['control'])]
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,threshold=threshold,development_passed=selection['development_passed'],conditional_thresholds=thresholds,conditional_development=conditional_dev,model_sha256=sha(mp),metrics=metrics,cohorts=cohorts,reports=reports,passed=passed,selected=max(passed,key=lambda a:metrics[a]['score']) if passed else 'control',seconds=time.monotonic()-start,leaderboard_submitted=False,scope=cfg['protocol']),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
