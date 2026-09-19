"""Audit the exact candidate capture and paired CPU smoke result."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify
from audit_cpu_control import aggregate
def main():
    gpu=ROOT/'outputs/e022_recovery';cpu=ROOT/'outputs/e022_cpu_recovery';root=cpu/'visual_exact'
    a=verify(gpu,'kaggle/visual_capture','results/E022_launch.json','visual_capture_package')
    b=verify(cpu,'kaggle/visual_exact','results/E022_CPU_launch.json','visual_exact_package')
    cr=read(gpu/'visual_capture/result.json');r=read(root/'result.json');assert cr['status']==r['status']=='complete'
    names=read(ROOT/'baseline/e022_capture.json')['videos'];assert cr['videos']==names
    frozen=read(root/'frozen_predictions.json');assert frozen['videos']==names and not frozen['annotations_used']
    old={v['dataset']:v for v in read(ROOT/'outputs/e017_compare_recovery/calibration_compare/harmonic_metrics.json')['samples']}
    metrics={arm:read(root/(arm+'_metrics.json')) for arm in ['harmonic','visual','sequence']};paired={}
    for arm,m in metrics.items():
        assert sorted(v['dataset'] for v in m['samples'])==sorted(names)
        assert m['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        computed=aggregate(m['samples']);assert all(math.isclose(v,m['summary'][k],rel_tol=0,abs_tol=1e-12) for k,v in computed.items())
        assert m['summary']==r['summaries'][arm]
        if arm=='harmonic':
            assert all(row==old[row['dataset']] for row in m['samples'])
        else:
            paired[arm]=[dict(dataset=row['dataset'],adj_edge_delta=row['adj_edge_jaccard']-old[row['dataset']]['adj_edge_jaccard'],
                tp_delta=row['edge_tp']-old[row['dataset']]['edge_tp'],fp_delta=row['edge_fp']-old[row['dataset']]['edge_fp']) for row in m['samples']]
    reproduction=read(ROOT/'results/E022_control_reproduction.json');assert reproduction['all_edges_equal'] and reproduction['all_node_coordinates_and_ids_equal']
    receipt=dict(status='complete',gpu_source_files_verified=a,cpu_source_files_verified=b,gpu_process_seconds=cr['seconds'],cpu_seconds=r['seconds'],
        captured_pairs=cr['pairs'],control_exactly_reproduced=True,summaries=r['summaries'],deltas=r['deltas'],paired=paired,
        result_sha256=sha(root/'result.json'),aggregates_recomputed=True,scope=r['scope'],leaderboard_submitted=False,
        decision='Freeze visual-only configuration and validate on eight filename-selected reserved videos before any submission')
    (ROOT/'results/E022_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
