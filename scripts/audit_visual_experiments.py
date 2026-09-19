"""Check frozen sources, paired cohorts and independently recompute metrics."""
import argparse,json,math
from audit_e017_completed import ROOT,read,sha,verify
from audit_cpu_control import aggregate

def main(experiment):
    stem='visual_sequence' if experiment=='E020' else 'visual_replay'
    folder=ROOT/('outputs/'+experiment.lower()+'_recovery');root=folder/stem
    launch='results/'+experiment+('_launch_v3.json' if experiment=='E020' else '_launch.json')
    count=verify(folder,'kaggle/'+stem,launch,stem+'_package')
    assert read(folder/'status.json')['status']=='COMPLETE'
    result=read(root/'result.json');assert result['status']=='complete'
    assert not read(root/'frozen_predictions.json')['annotations_used']
    baseline=read(ROOT/'outputs/e017_compare_recovery/calibration_compare/harmonic_metrics.json')
    assert result['harmonic']==baseline['summary']
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    reference={r['dataset']:r for r in baseline['samples']};arms={}
    for arm in ('visual','sequence'):
        metric=read(root/(arm+'_metrics.json'));assert sorted(r['dataset'] for r in metric['samples'])==sorted(names)
        assert metric['official_commit']==baseline['official_commit']
        computed=aggregate(metric['samples'])
        assert all(math.isclose(v,metric['summary'][k],abs_tol=1e-12,rel_tol=0) for k,v in computed.items())
        assert metric['summary']==result['arms'][arm]['summary']
        paired=[]
        for r in metric['samples']:
            b=reference[r['dataset']];assert r['num_pred_nodes']==b['num_pred_nodes']
            paired.append(dict(dataset=r['dataset'],adj_edge_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard'],tp_delta=r['edge_tp']-b['edge_tp'],fp_delta=r['edge_fp']-b['edge_fp']))
        reports=read(root/(arm+'_reports.json'))
        for r in reports.values():
            if 'final_energy' in r:assert r['final_energy']>=r['initial_energy']-1e-7
        arms[arm]=dict(summary=computed,score_delta=computed['score']-baseline['summary']['score'],paired=paired,
            better=sum(r['adj_edge_delta']>0 for r in paired),worse=sum(r['adj_edge_delta']<0 for r in paired),
            added=sum(r.get('added',0) for r in reports.values()),removed=sum(r.get('removed',0) for r in reports.values()),
            alternatives=sum(r.get('alternatives',0) for r in reports.values()))
    receipt=dict(experiment=experiment,status='complete',source_files_verified=count,arms=arms,seconds=result['seconds'],
        result_sha256=sha(root/'result.json'),aggregates_recomputed=True,unchanged_node_counts=True,csv_downloaded_and_verified=False,
        official_metric_rerun_locally=False,leaderboard_submitted=False,scope=result['scope'])
    (ROOT/'results'/(experiment+'_completed.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='arms'}));print(json.dumps({a:{k:v for k,v in x.items() if k!='paired'} for a,x in arms.items()},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment',choices=['E020','E021']);args=parser.parse_args();main(args.experiment)
