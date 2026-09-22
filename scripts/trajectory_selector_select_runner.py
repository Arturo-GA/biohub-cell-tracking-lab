"""CPU sparse supervision, fitted selector and whole-graph development selection."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
import tracksdata as td
from residual_detector_io import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from biohub_lab.neural_complement import neural_chains
from biohub_lab.detection_identity import maximum_vote_edges
from biohub_lab.residual_tracklets import SCALE,augment
from biohub_lab.trajectory_selector import proposals,integrate,fit,predict,usefulness_label
from biohub_official.metrics import _evaluate,_evaluate_matched_graph,summarise

def supervised_edges(nodes,edges,truth):
    k=td.DEFAULT_ATTR_KEYS;g=td.graph.InMemoryGraph()
    for a in ('z','y','x'):g.add_node_attr_key(a,pl.Float64,0.)
    keys=sorted(nodes);ids=g.bulk_add_nodes([dict(t=nodes[n]['t'],**{a:float(nodes[n][a]) for a in ('z','y','x')}) for n in keys]);mapping=dict(zip(keys,ids));reverse=dict(zip(ids,keys))
    g.bulk_add_edges([dict(source_id=mapping[a],target_id=mapping[b]) for a,b in edges]);_evaluate(g,truth,'jaccard',tuple(SCALE),7.)
    labels={};correct={}
    matched={r[k.NODE_ID]:r[k.MATCHED_NODE_ID] for r in g.node_attrs(attr_keys=[k.NODE_ID,k.MATCHED_NODE_ID]).iter_rows(named=True)}
    for r in _evaluate_matched_graph(g,truth).iter_rows(named=True):
        pair=(reverse[r[k.EDGE_SOURCE]],reverse[r[k.EDGE_TARGET]])
        if r['pred_valid']:labels[pair]=int(r[k.MATCHED_EDGE_MASK])
        if r[k.MATCHED_EDGE_MASK]:correct[pair]=(int(matched[r[k.EDGE_SOURCE]]),int(matched[r[k.EDGE_TARGET]]))
    return labels,correct

def write_graphs(path,graphs):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(COLUMNS);i=0
        for video,(nodes,edges) in graphs.items():
            for k in sorted(nodes):w.writerow([i,video,'node',k,*[nodes[k][a] for a in ('t','z','y','x')],-1,-1]);i+=1
            for a,b in edges:w.writerow([i,video,'edge',-1,-1,-1,-1,-1,a,b]);i+=1

def counts(m):
    return dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})

def check_sparse_label_semantics():
    truth=td.graph.InMemoryGraph()
    for a in ('z','y','x'):truth.add_node_attr_key(a,pl.Float64,0.)
    ids=truth.bulk_add_nodes([dict(t=t,z=0.,y=0.,x=0.) for t in (0,1)])
    truth.bulk_add_edges([dict(source_id=ids[0],target_id=ids[1])])
    nodes={i:dict(t=t,z=0,y=0,x=x) for i,(t,x) in enumerate([(0,0),(1,0),(0,100),(1,100)])}
    labels,_=supervised_edges(nodes,[(0,1),(0,3),(2,3)],truth)
    assert labels=={(0,1):1,(0,3):0},labels

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e063_protocol.json').read_text());pins=json.loads((package/'baseline/e063_select_pins.json').read_text())
    out=Path('/kaggle/working/trajectory_selector');out.mkdir(exist_ok=True)
    up=one('selector_inputs/result.json');pp=one('selector_associations/result.json');assert sha(up)==pins['PREPARE'] and sha(pp)==pins['INFER'];union=json.loads(up.read_text());pred=json.loads(pp.read_text())
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    check_sparse_label_semantics()
    examples=[];labels=[];weights=[];reports=[];development={}
    for u in union['records']:
        video=u['video'];r=next(r for r in pred['records'] if r['video']==video);assert sha(up.parent/u['file'])==u['sha256']==r['union_sha256'] and sha(pp.parent/r['file'])==r['sha256']
        with np.load(up.parent/u['file']) as z:coords,mapping,confidence=z['coords'],z['mapping'],z['prob']
        with np.load(pp.parent/r['file']) as z:edges,prob=z['edges'],z['prob']
        old=np.flatnonzero(mapping>=0);nodes={int(mapping[i]):dict(zip(('t','z','y','x'),map(int,coords[i]))) for i in old}
        keep=(mapping[edges]>=0).all(1)&(prob>.05)&(np.linalg.norm((coords[edges[:,1],1:]-coords[edges[:,0],1:])*SCALE,axis=1)<=14)
        chosen=maximum_vote_edges(coords,edges[keep],np.log(prob[keep]/.05));base=sorted(map(tuple,mapping[chosen].tolist()))
        paths=neural_chains(coords,edges,prob);items=proposals(coords,mapping,confidence,paths,edges,prob)
        if video in cfg['development']:
            development[video]=(nodes,base,coords,mapping,confidence,paths,items);continue
        assert video in cfg['fit'] and video not in cfg['evaluation']
        ns,es,report=integrate(nodes,base,coords,mapping,items,np.ones(len(items)),0,cap_fraction=None)
        truth=td.graph.IndexedRXGraph.from_geff(str(train/(video+'.geff')))[0]
        _,original_correct=supervised_edges(nodes,base,truth);already=set(original_correct.values())
        known,correct=supervised_edges(ns,es,truth);n0=n1=unknown=redundant=0;local=[]
        for p in report['accepted']:
            values=[known[e] for e in p['edges'] if e in known]
            label=usefulness_label(p['edges'],known,correct,already)
            if label is None:unknown+=1;continue
            novel=any(e in correct and correct[e] not in already for e in p['edges'])
            redundant+=int(all(values) and not novel)
            n1+=label;n0+=1-label
            local.append((items[p['index']]['x'],label,float(np.sqrt(len(values)))))
        norm=sum(w for _,_,w in local)
        for x,y,w in local:examples.append(x);labels.append(y);weights.append(w/norm)
        reports.append(dict(video=video,positive=n1,negative=n0,redundant_known=redundant,unknown_excluded=unknown,proposals=len(items),feasible=len(report['accepted'])))
        print('ENSEMBLE_SELECTOR_LABELS',reports[-1],flush=True)
    np.savez_compressed(out/'fit_examples.npz',x=examples,y=labels,weight=weights)
    (out/'label_reports.json').write_text(json.dumps(reports,indent=2))
    model=fit(examples,labels,weights);model_path=out/'model.json';model_path.write_text(json.dumps(model));model_sha=sha(model_path)
    print('ENSEMBLE_SELECTOR_FIT',len(labels),sum(labels),'positive',model['iterations'],'iterations',flush=True)
    # Freeze the fit before any development metric is read.
    metrics={};cohorts={};graphs={'control':{},'rule':{},**{f'learned_{t}':{} for t in cfg['thresholds']}}
    for video,(nodes,base,coords,mapping,confidence,paths,items) in development.items():
        p=predict(model,[v['x'] for v in items]);graphs['control'][video]=(nodes,base)
        ns,es,_=augment(nodes,base,coords,confidence,np.zeros((len(coords),1)),True,paths,mapping);graphs['rule'][video]=(ns,es)
        for threshold in cfg['thresholds']:
            ns,es,_=integrate(nodes,base,coords,mapping,items,p,threshold,cfg['node_budget_fraction']);graphs[f'learned_{threshold}'][video]=(ns,es)
    frozen={}
    for arm,gs in graphs.items():path=out/(arm+'.csv');write_graphs(path,gs);frozen[arm]=sha(path)
    (out/'frozen_development.json').write_text(json.dumps(dict(model_sha256=model_sha,graphs=frozen),indent=2))
    data=evaluation_dir(train,cfg['development'],out/'data')
    for arm in graphs:
        m=evaluate_csv(out/(arm+'.csv'),data);metrics[arm]=counts(m);cohorts[arm]={e:summarise([s for s in m['samples'] if s['dataset'].startswith(e)]) for e in ('44b6','6bba')}
        (out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));assert sha(out/(arm+'.csv'))==frozen[arm]
        print('ENSEMBLE_SELECTOR_DEV',arm,metrics[arm],flush=True)
    valid=[a for a in graphs if a.startswith('learned_') and metrics[a]['score']>metrics['control']['score']+1e-6 and metrics[a]['edge_tp']>metrics['control']['edge_tp'] and metrics[a]['division_fp']<=metrics['control']['division_fp'] and all(cohorts[a][e]['score']>=cohorts['control'][e]['score'] for e in cohorts[a])]
    selected=max(valid,key=lambda a:metrics[a]['score']) if valid else 'control';threshold=float(selected.split('_',1)[1]) if valid else 1.01
    assert sha(model_path)==model_sha
    diagnostic=[a for a in graphs if a.startswith('learned_') and frozen[a]!=frozen['control']]
    best=max(diagnostic,key=lambda a:metrics[a]['score']) if diagnostic else 'learned_1.01'
    result=dict(status='complete',config=cfg,sparse_label_selftest_passed=True,model_converged=model['converged'],model_iterations=model['iterations'],model='model.json',model_sha256=model_sha,selected=selected,threshold=threshold,diagnostic_threshold=float(best.split('_',1)[1]),development_passed=bool(valid),metrics=metrics,cohorts=cohorts,reports=reports,fit_examples=len(labels),positive=int(sum(labels)),negative=len(labels)-int(sum(labels)),seconds=time.monotonic()-start,leaderboard_submitted=False,scope=cfg['protocol'])
    (out/'result.json').write_text(json.dumps(result,indent=2));print('ENSEMBLE_SELECTOR_SELECTED',selected,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
