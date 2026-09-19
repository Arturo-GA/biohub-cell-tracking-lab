"""Verify full paired comparison against source payload and sample counts."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify
from audit_cpu_control import aggregate
def main():
    folder=ROOT/'outputs/e017_compare_recovery';root=folder/'calibration_compare'
    count=verify(folder,'kaggle/calibration_compare','results/E017_compare_launch.json','calibration_compare_package')
    result=read(root/'result.json');assert result['status']=='complete' and result['accelerator']=='none'
    assert read(folder/'status.json')['status']=='COMPLETE'
    pins=read(ROOT/'baseline/e017_compare_pins.json');export=read(root/'export_receipt.json')
    assert export['source_sha256']==pins['control_csv_sha256']==export['validated_sha256']
    assert not export['changed_spatial_coordinates'] and not export['topology_changed']
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    paths=dict(harmonic=root/'harmonic_metrics.json',tissue=ROOT/'outputs/e017_recovery/tissue_trajectory/candidate_metrics.json',
        learned=ROOT/'outputs/e016_recovery/identity_parent/learned_metrics.json',geometric=ROOT/'outputs/e016_recovery/identity_parent/geometric_metrics.json')
    metrics={};totals={}
    for label,path in paths.items():
        if label!='harmonic':assert sha(path)==pins[label+'_metrics_sha256']
        m=read(path);assert sorted(r['dataset'] for r in m['samples'])==sorted(names)
        assert m['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        assert m['summary']==result['summaries'][label]
        assert all(math.isclose(v,m['summary'][k],rel_tol=0,abs_tol=1e-12) for k,v in aggregate(m['samples']).items())
        metrics[label]={r['dataset']:r for r in m['samples']}
        totals[label]={k:sum(r[k] for r in m['samples']) for k in ['edge_tp','edge_fp','edge_fn','num_pred_nodes']}
    assert totals['harmonic']['num_pred_nodes']==export['nodes']
    paired=read(root/'paired.json');counts={}
    for label,rows in paired.items():
        assert sorted(r['dataset'] for r in rows)==sorted(names)
        assert math.isclose(result['score_delta_vs_harmonic'][label],result['summaries'][label]['score']-result['summaries']['harmonic']['score'],abs_tol=1e-12)
        for r in rows:
            a=metrics[label][r['dataset']];b=metrics['harmonic'][r['dataset']]
            assert math.isclose(r['adj_edge_delta'],a['adj_edge_jaccard']-b['adj_edge_jaccard'],abs_tol=1e-12)
            for k in ['tp','fp','fn']:assert r[k+'_delta']==a['edge_'+k]-b['edge_'+k]
        counts[label]=dict(better=sum(r['adj_edge_delta']>0 for r in rows),worse=sum(r['adj_edge_delta']<0 for r in rows))
    receipt=dict(status='complete',source_files_verified=count,seconds=result['seconds'],summaries=result['summaries'],
        score_delta_vs_harmonic=result['score_delta_vs_harmonic'],paired_videos=counts,totals=totals,
        export_unchanged=True,result_sha256=sha(root/'result.json'),aggregates_recomputed=True,
        csv_downloaded_and_verified=False,official_metric_rerun_locally=False,leaderboard_submitted=False,
        decision='Reject E016/E017 as full replacements; retain full Harmonic. Diagnose complementary correct links before any fusion.',
        scope=result['scope'])
    (ROOT/'results/E017_compare_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
