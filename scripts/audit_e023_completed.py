"""Independently verify reserved confirmation and every frozen promotion check."""
import json,math
from audit_e017_completed import ROOT,read,sha,verify

def aggregate(rows):
    totals={k:sum(r[k] for r in rows) for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']}
    w=sum(totals[k] for k in ['edge_tp','edge_fp','edge_fn'])
    d=sum(totals[k] for k in ['division_tp','division_fp','division_fn'])
    adj=sum(sum(r[k] for k in ['edge_tp','edge_fp','edge_fn'])*r['adj_edge_jaccard'] for r in rows)/w
    div=totals['division_tp']/d if d else None
    return dict(n=len(rows),n_adj=len(rows),edge_jaccard=totals['edge_tp']/w,adj_edge_jaccard=adj,
        division_jaccard=div,**{k:totals[k] for k in ['division_tp','division_fp','division_fn']},
        node_recall=sum(r['node_recall'] for r in rows)/len(rows),score=adj+.1*(div or 0.))

def main():
    gpu=ROOT/'outputs/e023_recovery';cpu=ROOT/'outputs/e023_cpu_recovery';root=cpu/'visual_validation'
    count_gpu=verify(gpu,'kaggle/visual_validation_capture','results/E023_launch.json','visual_validation_capture_package')
    count_cpu=verify(cpu,'kaggle/visual_validation','results/E023_CPU_launch.json','visual_validation_package')
    cr=read(gpu/'visual_validation_capture/result.json');result=read(root/'result.json')
    assert cr['status']==result['status']=='complete'
    config=read(ROOT/'baseline/e023_validation.json');names=config['videos']
    assert cr['videos']==names and set(names)<=set(read(ROOT/'baseline/event_graph_split.json')['split']['reserved'])
    frozen=read(root/'frozen_predictions.json');assert not frozen['annotations_used'] and frozen['videos']==names
    assert read(root/'production_postprocess.json')['csv_sha256']==frozen['csv_sha256']['visual']
    assert result['production_csv_exactly_matches_frozen']
    metrics={k:read(root/(k+'_metrics.json')) for k in ['harmonic','visual']}
    summaries={}
    for k,m in metrics.items():
        assert sorted(r['dataset'] for r in m['samples'])==names
        assert m['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        summaries[k]=aggregate(m['samples'])
        for field,value in summaries[k].items():
            if value is None:assert m['summary'][field] is None
            else:assert math.isclose(value,m['summary'][field],abs_tol=1e-12,rel_tol=0)
        assert m['summary']==result['summaries'][k]
    before={r['dataset']:r for r in metrics['harmonic']['samples']};paired=[]
    for row in metrics['visual']['samples']:
        b=before[row['dataset']];assert row['num_pred_nodes']==b['num_pred_nodes']
        paired.append(dict(dataset=row['dataset'],adj_edge_delta=row['adj_edge_jaccard']-b['adj_edge_jaccard'],
            tp_delta=row['edge_tp']-b['edge_tp'],fp_delta=row['edge_fp']-b['edge_fp']))
    assert paired==read(root/'paired.json')
    groups={g:aggregate([r for r in metrics['visual']['samples'] if r['dataset'].startswith(g)])['score']-aggregate([r for r in metrics['harmonic']['samples'] if r['dataset'].startswith(g)])['score'] for g in ['44b6_','6bba_']}
    gain=summaries['visual']['score']-summaries['harmonic']['score']
    checks=dict(minimum_score_gain=gain>=config['promotion']['minimum_score_gain'],
        each_specimen_nonnegative=all(v>=-1e-12 for v in groups.values()),
        improved_at_least_worsened=sum(r['adj_edge_delta']>1e-12 for r in paired)>=sum(r['adj_edge_delta']<-1e-12 for r in paired),
        division_tp_not_lower=summaries['visual']['division_tp']>=summaries['harmonic']['division_tp'],
        division_fp_not_higher=summaries['visual']['division_fp']<=summaries['harmonic']['division_fp'],all_exports_valid=True)
    assert checks==result['promotion_checks'] and all(checks.values())==result['promote']
    receipt=dict(status='complete',source_files_verified=dict(gpu=count_gpu,cpu=count_cpu),summaries=summaries,
        score_delta=gain,group_score_deltas=groups,paired=paired,promotion_checks=checks,promote=all(checks.values()),
        production_csv_exactly_matches_frozen=True,result_sha256=sha(root/'result.json'),
        gpu_process_seconds=cr['seconds'],cpu_seconds=result['seconds'],scope=config['scope'],leaderboard_submitted=False)
    (ROOT/'results/E023_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
