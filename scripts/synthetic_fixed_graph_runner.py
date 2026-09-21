"""E057: isolate association gains on exactly the old nodes and divisions."""
import csv,json,sys,time
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

def raw_nodes(root,name):
    paths=list((root/'harmonic_control/tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'));assert len(paths)==1
    graph=td.graph.IndexedRXGraph.from_geff(paths[0]);graph=graph[0] if isinstance(graph,tuple) else graph
    return {int(r['node_id']):{a:float(r[a]) for a in ('t','z','y','x')} for r in graph.node_attrs().iter_rows(named=True)}

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/synthetic_fixed_graph');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e057_protocol.json').read_text());p=one('visual_validation/result.json');old=p.parent
    captures={}
    input_root=Path('/kaggle/input')
    parents=[input_root,*input_root.glob('*'),*input_root.glob('*/*'),*input_root.glob('*/*/*')]
    paths=list(dict.fromkeys(p/'visual_validation_capture/result.json' for p in parents if (p/'visual_validation_capture/result.json').is_file()))
    for path in paths:
        r=json.loads(path.read_text());assert r['status']=='complete';captures[r['experiment']]=path.parent
    assert set(captures)=={'E023-capture','E056'};control=captures['E023-capture'];synthetic=captures['E056']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data);graphs=read_and_validate(old/'harmonic.csv',shapes)
    assert sha(old/'harmonic.csv')==json.loads((old/'frozen_predictions.json').read_text())['csv_sha256']['harmonic']
    arms=['visual','specialist','balanced','uncertainty'];handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};ids={a:0 for a in arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for name in cfg['videos']:
            nodes,original=graphs[name];old_raw=raw_nodes(control,name)
            frames=[]
            for path in sorted((synthetic.parent/'visual_candidate_cache'/name).glob('*.npz')):
                with np.load(path,allow_pickle=False) as values:frame={k:values[k] for k in values.files}
                with np.load(control.parent/'visual_candidate_cache'/name/path.name,allow_pickle=False) as previous:
                    for key in ('source_coords','target_coords'):
                        assert np.array_equal(frame[key],previous[key]),'Detection coordinates changed; cannot claim fixed detections'
                frames.append(frame)
            assert len(frames)==shapes[name][0]-1
            # Coordinates map to the OLD GEFF IDs. IDs in the new ILP graph may be renumbered.
            fresh,audit=decode_cache(old_raw,nodes,frames)
            with np.load(old/(name+'_visual.npz')) as z:visual={tuple(map(int,e)):float(v) for e,v in zip(z['edges'],z['prob'])}
            for arm in arms:
                probs,report=(visual,{}) if arm=='visual' else pool(visual,fresh,arm)
                edges,assignment=assign(nodes,original,probs)
                for k in sorted(nodes):writers[arm].writerow([ids[arm],name,'node',k,*[nodes[k][a] for a in ('t','z','y','x')],-1,-1]);ids[arm]+=1
                for a,b in edges:writers[arm].writerow([ids[arm],name,'edge',-1,-1,-1,-1,-1,a,b]);ids[arm]+=1
                reports.append(dict(video=name,arm=arm,weights=report,assignment=assignment,remap=audit))
            print('ENSEMBLE_SYNTH_FIXED',name,flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes))
    metrics={}
    for arm in arms:
        read_and_validate(out/(arm+'.csv'),shapes);m=evaluate_csv(out/(arm+'.csv'),data);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        assert sha(out/(arm+'.csv'))==hashes[arm];print('ENSEMBLE_SYNTH_FIXED_METRIC',arm,metrics[arm],flush=True)
    assert abs(metrics['visual']['score']-.9502832669356378)<1e-10
    selected=max(arms,key=lambda a:metrics[a]['score'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,reports=reports,selected=selected,improved_over_visual=metrics[selected]['score']>metrics['visual']['score']+1e-8,seconds=time.monotonic()-start,scope='Eight repeatedly used videos, exploratory; no independent holdout',leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
