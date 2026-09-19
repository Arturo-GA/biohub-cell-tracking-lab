"""E018: official-matcher overlap of frozen Harmonic and tissue predictions, CPU."""
from pathlib import Path
from collections import Counter
import time,traceback
import torch
from association_cpu_runner import locate,sha,read,save,evaluation_dir
from biohub_lab.submission import read_and_validate
from biohub_lab.evaluate import shapes_for
from biohub_lab.edge_complementarity import overlap,feature_bins

def matched_edges(nodes,edges,truth,expected):
    import tracksdata as td
    import polars as pl
    from biohub_official.metrics import _evaluate,_evaluate_matched_graph
    k=td.DEFAULT_ATTR_KEYS;graph=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):graph.add_node_attr_key(axis,pl.Float64,0.)
    keys=sorted(nodes);ids=graph.bulk_add_nodes([{'t':nodes[n]['t'],**{axis:float(nodes[n][axis]) for axis in ('z','y','x')}} for n in keys])
    mapping=dict(zip(keys,ids));reverse={v:n for n,v in mapping.items()}
    graph.bulk_add_edges([{'source_id':mapping[a],'target_id':mapping[b]} for a,b in edges])
    _evaluate(graph,truth,'jaccard',(1.625,.40625,.40625),7.)
    attrs=graph.node_attrs(attr_keys=[k.NODE_ID,k.MATCHED_NODE_ID])
    matches={r[k.NODE_ID]:r[k.MATCHED_NODE_ID] for r in attrs.iter_rows(named=True)}
    table=_evaluate_matched_graph(graph,truth);tp=set();classified={};fp=0
    for row in table.iter_rows(named=True):
        a,b=row[k.EDGE_SOURCE],row[k.EDGE_TARGET];edge=(reverse[a],reverse[b])
        if row[k.MATCHED_EDGE_MASK]:
            pair=(int(matches[a]),int(matches[b]));tp.add(pair);classified[edge]=('tp',pair)
        elif row['pred_valid']:fp+=1;classified[edge]=('fp',None)
        else:classified[edge]=('unknown',None)
    assert len(tp)==expected['edge_tp'] and fp==expected['edge_fp']
    assert truth.num_edges()-len(tp)==expected['edge_fn']
    return tp,{v for v in matches.values() if v is not None and v!=-1},classified

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/edge_complementarity')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E018',status='verifying',accelerator='none',training_started=False,leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU only')
        import tracksdata as td
        pins=read(package/'baseline/e018_pins.json')
        compare=locate(input_root,'calibration_compare/result.json').parent
        tissue=locate(input_root,'tissue_trajectory/result.json').parent
        for path,key in [(compare/'result.json','compare_result'),(tissue/'result.json','tissue_result'),
            (compare/'harmonic_validated.csv','harmonic_csv'),(tissue/'candidate.csv','tissue_csv'),
            (compare/'harmonic_metrics.json','harmonic_metrics'),(tissue/'candidate_metrics.json','tissue_metrics')]:
            assert sha(path)==pins[key+'_sha256'],key
        names=read(package/'baseline/event_graph_split.json')['split']['calibration']
        train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
        data=evaluation_dir(train,names,root/'calibration_data');shapes=shapes_for(data)
        predictions=dict(harmonic=read_and_validate(compare/'harmonic_validated.csv',shapes),tissue=read_and_validate(tissue/'candidate.csv',shapes))
        metrics={label:{r['dataset']:r for r in read(path)['samples']} for label,path in [('harmonic',compare/'harmonic_metrics.json'),('tissue',tissue/'candidate_metrics.json')]}
        reports={};histograms={}
        for name in names:
            truth=td.graph.IndexedRXGraph.from_geff(str(data/(name+'.geff')))[0];k=td.DEFAULT_ATTR_KEYS
            gt_edges={(int(r[k.EDGE_SOURCE]),int(r[k.EDGE_TARGET])) for r in truth.edge_attrs(attr_keys=[k.EDGE_SOURCE,k.EDGE_TARGET]).iter_rows(named=True)}
            recovered={}
            # Geometry is computed without labels; classification labels below are diagnostic only.
            bins=feature_bins(*predictions['tissue'][name])
            for label in ['harmonic','tissue']:
                recovered[label]=matched_edges(*predictions[label][name],truth,metrics[label][name])
            h,hn,_=recovered['harmonic'];t,_,classified=recovered['tissue']
            reports[name]=overlap(h,t,gt_edges,hn);hist={key:Counter() for key in ['rescued_tp','shared_tp','false_positive','unknown']}
            for edge,(status,pair) in classified.items():
                group=('rescued_tp' if pair not in h else 'shared_tp') if status=='tp' else 'false_positive' if status=='fp' else 'unknown'
                hist[group].update(bins[edge])
            histograms[name]={key:dict(value) for key,value in hist.items()}
            save(root/'overlap_by_video.json',reports);save(root/'feature_histograms.json',histograms)
            print('COMPLEMENTARITY_READY',name,reports[name],flush=True)
        keys=[key for key,v in reports[names[0]].items() if isinstance(v,int)]
        totals={key:sum(r[key] for r in reports.values()) for key in keys}
        result.update(status='complete',seconds=time.monotonic()-start,videos=len(names),totals=totals,
            official_tp_fp_counts_reproduced=True,no_fused_predictions_created=True,evaluation_cohort_used=False,
            scope='GT-dependent diagnostic on 16 calibration videos. Oracle edge union is not a score or deployable predictor.',
            next_action='Inspect rescued links versus shared links and false positives; do not train/select a gate on these calibration labels')
        save(root/'result.json',result);print('COMPLEMENTARITY_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
