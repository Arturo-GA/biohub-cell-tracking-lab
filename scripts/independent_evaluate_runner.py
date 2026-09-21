"""E059: standalone synthetic experts and fixed-weight ensembles on original nodes."""
import csv,json,sys,time,math
from pathlib import Path
import numpy as np
import tracksdata as td
from association_cpu_runner import evaluation_dir
from ensemble_evaluate_runner import one,sha
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_lab.visual_candidates import decode_cache
from biohub_lab.visual_assignment import assign
from biohub_lab.association_pool import pool
from residual_audit_runner import match

def raw_nodes(root,name):
    paths=list((root/'harmonic_control/tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'));assert len(paths)==1
    graph=td.graph.IndexedRXGraph.from_geff(paths[0]);graph=graph[0] if isinstance(graph,tuple) else graph
    return {int(r['node_id']):{a:float(r[a]) for a in ('t','z','y','x')} for r in graph.node_attrs().iter_rows(named=True)}

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/independent_evaluation');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e059_protocol.json').read_text());p=one('visual_validation/result.json');old=p.parent
    control=one('visual_validation_capture/result.json').parent
    capture=one('independent_experts/result.json').parent
    captured=json.loads((capture/'result.json').read_text());assert captured['status']=='complete'
    manifest={(r['video'],r['model']):r['files'] for r in captured['records']}
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data);graphs=read_and_validate(old/'harmonic.csv',shapes)
    assert sha(old/'harmonic.csv')==json.loads((old/'frozen_predictions.json').read_text())['csv_sha256']['harmonic']
    arms=['visual','swa','fold0','swa_half','fold0_half','three_way'];handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};ids={a:0 for a in arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for name in cfg['videos']:
            nodes,original=graphs[name];old_raw=raw_nodes(control,name)
            fresh={};audits={}
            for model in cfg['models']:
                frames=[]
                for path in sorted((capture/model/name).glob('*.npz')):
                    assert sha(path)==manifest[name,model][path.name]
                    with np.load(path,allow_pickle=False) as values:frame={k:values[k] for k in values.files}
                    with np.load(control.parent/'visual_candidate_cache'/name/path.name,allow_pickle=False) as previous:
                        for key in ('source_coords','target_coords'):assert np.array_equal(frame[key],previous[key])
                    frames.append(frame)
                assert len(frames)==shapes[name][0]-1
                fresh[model],audits[model]=decode_cache(old_raw,nodes,frames)
            with np.load(old/(name+'_visual.npz')) as z:visual={tuple(map(int,e)):float(v) for e,v in zip(z['edges'],z['prob'])}
            assert all(set(visual)<=set(fresh[m]) for m in fresh), 'Every old candidate must retain an exact probability'
            for arm in arms:
                report={}
                if arm=='visual':probs=visual
                elif arm in fresh:probs=fresh[arm]
                elif arm.endswith('_half'):probs,report=pool(visual,fresh[arm[:-5]],'balanced')
                else:probs={e:math.exp(.5*math.log(max(p,1e-8))+.25*math.log(max(fresh['swa'][e],1e-8))+.25*math.log(max(fresh['fold0'][e],1e-8))) for e,p in visual.items()}
                edges,assignment=assign(nodes,original,probs)
                for k in sorted(nodes):writers[arm].writerow([ids[arm],name,'node',k,*[nodes[k][a] for a in ('t','z','y','x')],-1,-1]);ids[arm]+=1
                for a,b in edges:writers[arm].writerow([ids[arm],name,'edge',-1,-1,-1,-1,-1,a,b]);ids[arm]+=1
                reports.append(dict(video=name,arm=arm,weights=report,assignment=assignment,remap=audits))
            print('ENSEMBLE_INDEPENDENT',name,flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes))
    metrics={}
    for arm in arms:
        read_and_validate(out/(arm+'.csv'),shapes);m=evaluate_csv(out/(arm+'.csv'),data);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        assert sha(out/(arm+'.csv'))==hashes[arm];print('ENSEMBLE_INDEPENDENT_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['visual']['score']-.9502832669356378)<1e-10
    graphs_all={arm:read_and_validate(out/(arm+'.csv'),shapes) for arm in arms}
    metric_all={arm:{r['dataset']:r for r in json.loads((out/(arm+'_metrics.json')).read_text())['samples']} for arm in arms}
    complement=[]
    for name in cfg['videos']:
        truth=td.graph.IndexedRXGraph.from_geff(str(data/(name+'.geff')))[0]
        correct={arm:match(*graphs_all[arm][name],truth,metric_all[arm][name])[0] for arm in arms}
        for arm in arms:complement.append(dict(video=name,arm=arm,rescued=sorted(correct[arm]-correct['visual']),lost=sorted(correct['visual']-correct[arm])))
    totals={arm:dict(rescued=sum(len(r['rescued']) for r in complement if r['arm']==arm),lost=sum(len(r['lost']) for r in complement if r['arm']==arm)) for arm in arms}
    print('ENSEMBLE_INDEPENDENT_COMPLEMENT',totals,flush=True)
    selected=max(arms,key=lambda a:metrics[a]['score'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,complementarity=totals,by_video=complement,reports=reports,selected=selected,improved_over_visual=metrics[selected]['score']>metrics['visual']['score']+1e-8,seconds=time.monotonic()-start,scope='Eight repeatedly used videos, exploratory; no independent holdout',leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
