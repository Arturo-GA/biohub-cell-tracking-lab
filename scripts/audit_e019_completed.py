"""Audit the anchored-tracklet trial and paired complete-Harmonic metrics."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify
from audit_cpu_control import aggregate
def main():
    folder=ROOT/'outputs/e019_recovery';root=folder/'anchored_tracklets'
    count=verify(folder,'kaggle/anchored_tracklets','results/E019_launch.json','anchored_tracklets_package')
    result=read(root/'result.json');assert result['status']=='complete'
    frozen=read(root/'frozen_predictions.json');names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    assert frozen['videos']==names and not frozen['calibration_annotations_read']
    before=read(ROOT/'outputs/e017_compare_recovery/calibration_compare/harmonic_metrics.json');after=read(root/'candidate_metrics.json')
    for m,label in [(before,'harmonic'),(after,'candidate')]:
        assert sorted(r['dataset'] for r in m['samples'])==sorted(names)
        assert m['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        assert m['summary']==result[label]
        assert all(math.isclose(v,m['summary'][k],rel_tol=0,abs_tol=1e-12) for k,v in aggregate(m['samples']).items())
    reports=read(root/'recovery_reports.json');assert set(reports)==set(names)
    for r in reports.values():assert r['reference_edges_preserved'] and r['division_parents_preserved'] and not r['annotations_used']
    assert result['added_nodes']==sum(r['added_nodes'] for r in reports.values())
    assert result['added_edges']==sum(r['added_edges'] for r in reports.values())
    totals={label:{k:sum(r[k] for r in m['samples']) for k in ['edge_tp','edge_fp','edge_fn','num_pred_nodes']} for label,m in [('harmonic',before),('candidate',after)]}
    assert totals['candidate']['num_pred_nodes']-totals['harmonic']['num_pred_nodes']==result['added_nodes']
    reference={r['dataset']:r for r in before['samples']};paired=[]
    for r in after['samples']:
        b=reference[r['dataset']];paired.append(dict(dataset=r['dataset'],adj_edge_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard'],
            tp_delta=r['edge_tp']-b['edge_tp'],fp_delta=r['edge_fp']-b['edge_fp']))
    receipt=dict(status='complete',source_files_verified=count,seconds=result['seconds'],candidate=result['candidate'],harmonic=result['harmonic'],
        score_delta=result['score_delta'],added_nodes=result['added_nodes'],added_edges=result['added_edges'],totals=totals,paired=paired,
        videos_better=sum(r['adj_edge_delta']>0 for r in paired),videos_worse=sum(r['adj_edge_delta']<0 for r in paired),
        result_sha256=sha(root/'result.json'),aggregates_recomputed=True,csv_downloaded_and_verified=False,
        official_metric_rerun_locally=False,leaderboard_submitted=False,
        decision=('Reject anchored recovery; retain Harmonic and do not sweep thresholds on calibration' if result['score_delta']<=0 else 'Require separate validation before promotion'),scope=result['scope'])
    (ROOT/'results/E019_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
