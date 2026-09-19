"""Verify exact TP overlap accounting and marginal feature histograms."""
import json
from collections import Counter
from audit_e017_completed import ROOT,read,sha,verify
def main():
    folder=ROOT/'outputs/e018_recovery';root=folder/'edge_complementarity'
    count=verify(folder,'kaggle/edge_complementarity','results/E018_launch.json','edge_complementarity_package')
    result=read(root/'result.json');assert result['status']=='complete' and result['official_tp_fp_counts_reproduced']
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    rows=read(root/'overlap_by_video.json');hist=read(root/'feature_histograms.json');assert set(rows)==set(hist)==set(names)
    hm={r['dataset']:r for r in read(ROOT/'outputs/e017_compare_recovery/calibration_compare/harmonic_metrics.json')['samples']}
    tm={r['dataset']:r for r in read(ROOT/'outputs/e017_recovery/tissue_trajectory/candidate_metrics.json')['samples']}
    totals=Counter();features={k:Counter() for k in ['rescued_tp','shared_tp','false_positive','unknown']}
    for name,r in rows.items():
        assert r['harmonic_tp']==hm[name]['edge_tp']==r['both_correct']+r['harmonic_only']
        assert r['tissue_tp']==tm[name]['edge_tp']==r['both_correct']+r['tissue_only']
        assert r['gt_edges']==sum(r[k] for k in ['both_correct','harmonic_only','tissue_only','neither_correct'])
        assert r['diagnostic_union_tp']==r['gt_edges']-r['neither_correct']
        assert r['tissue_only']==r['rescued_with_both_endpoints_matched_in_harmonic']+r['rescued_with_missing_harmonic_endpoint']
        totals.update({k:v for k,v in r.items() if isinstance(v,int)})
        for label,h in hist[name].items():
            n=sum(v for k,v in h.items() if k.startswith('distance_'))
            assert n==h.get('two_sided_context',0)+h.get('incomplete_or_branched_context',0)
            assert n==sum(v for k,v in h.items() if k.startswith('acceleration_'))
            if label=='rescued_tp':assert n==r['tissue_only']
            if label=='shared_tp':assert n==r['both_correct']
            if label=='false_positive':assert n==tm[name]['edge_fp']
            features[label].update(h)
    assert dict(totals)==result['totals']
    receipt=dict(status='complete',seconds=result['seconds'],source_files_verified=count,totals=dict(totals),
        feature_histograms={k:dict(v) for k,v in features.items()},result_sha256=sha(root/'result.json'),
        accounting_verified=True,official_matching_rerun_locally=False,leaderboard_submitted=False,
        scope=result['scope'],decision='Test annotation-free anchored donor tracklets while preserving Harmonic edges and divisions; do not fit a gate on calibration GT')
    (ROOT/'results/E018_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
