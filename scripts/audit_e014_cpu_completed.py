"""Audit completed CPU reports against frozen code, cohort, and official aggregates."""
import ast, base64, hashlib, io, json, math, zipfile
from collections import Counter
from pathlib import Path
from audit_cpu_control import aggregate

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder=ROOT/'outputs/e014_cpu_recovery';root=folder/'association_cpu'
    result=read(root/'result.json');state=read(folder/'status.json');launch=read(ROOT/'results/E014_CPU_launch.json')
    assert state['status']=='COMPLETE' and result['status']=='complete'
    assert result['accelerator']=='none' and result['training_started'] is False
    assert launch['enable_gpu'] is False and launch['enable_tpu'] is False
    nb=read(ROOT/'kaggle/association_cpu/notebook.ipynb');tree=ast.parse(''.join(nb['cells'][1]['source']))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(folder/'association_cpu_package'/name).read_bytes(),name
        source_count=len(archive.namelist())
    pins=read(ROOT/'baseline/e014_cpu_pins.json');split=read(ROOT/'baseline/event_graph_split.json')['split']
    assert read(root/'frozen_configuration.json')==pins
    selection=read(root/'decoding_selection.json');assert selection==result['calibration']
    assert selection['evaluation_annotations_read'] is False
    assert selection['thresholds']==pins['thresholds']
    summaries={};solver_stats={};candidate=None;no_fallback=[]
    for role,arms in [('calibration',['continuations','divisions']),('evaluation',[selection['selected_arm']])]:
        frozen=read(root/f'{role}_predictions_frozen.json')
        assert frozen['role_annotations_read'] is False
        assert frozen['configuration_sha256']==sha(root/'frozen_configuration.json')
        assert set(frozen['csv_sha256'])==set(arms)
        for arm in arms:
            metrics=read(root/f'{role}_{arm}_metrics.json')
            assert metrics['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
            assert sorted(r['dataset'] for r in metrics['samples'])==sorted(split[role])
            assert all(math.isfinite(v) for r in metrics['samples'] for v in r.values() if isinstance(v,(int,float)))
            summary=aggregate(metrics['samples'])
            assert all(math.isclose(v,metrics['summary'][k],rel_tol=0,abs_tol=1e-12) for k,v in summary.items())
            summaries[role+'_'+arm]=summary
            windows=[]
            for name in split[role]:
                solver=read(root/role/arm/name/'prediction.json')['solver']
                assert solver['fallback_windows']==sum(w['fallback'] for w in solver['windows'])
                if role=='evaluation' and solver['fallback_windows']==0:no_fallback.append(name)
                windows.extend(solver['windows'])
            solver_stats[role+'_'+arm]=dict(windows=len(windows),fallback_windows=sum(w['fallback'] for w in windows),
                statuses=dict(Counter(w['status'] for w in windows)))
            if role=='calibration':assert summary==selection['summaries'][arm]
            else:
                candidate=metrics
                assert summary==result['candidate']
                assert frozen['selected_arm']==selection['selected_arm']
                assert frozen['csv_sha256'][arm]==result['candidate_csv_sha256']
    assert max(['continuations','divisions'],key=lambda a:selection['summaries'][a]['score'])==selection['selected_arm']
    control=read(ROOT/'outputs/cpu_finished_check/control/control_cpu/control_metrics.json')
    control_summary=aggregate(control['samples'])
    assert control_summary['score']==result['control_score']==pins['paired_control_score']
    assert math.isclose(result['score_delta'],result['candidate']['score']-control_summary['score'],abs_tol=1e-12)
    reference={r['dataset']:r for r in control['samples']}
    paired=dict(videos_better_adjusted_edges=sum(r['adj_edge_jaccard']>reference[r['dataset']]['adj_edge_jaccard'] for r in candidate['samples']),
        videos_worse_adjusted_edges=sum(r['adj_edge_jaccard']<reference[r['dataset']]['adj_edge_jaccard'] for r in candidate['samples']),
        candidate_nodes=sum(r['num_pred_nodes'] for r in candidate['samples']),control_nodes=sum(r['num_pred_nodes'] for r in control['samples']),
        videos_without_fallback=len(no_fallback),
        videos_without_fallback_worse_than_control=sum(r['dataset'] in no_fallback and r['adj_edge_jaccard']<reference[r['dataset']]['adj_edge_jaccard'] for r in candidate['samples']))
    paired['candidate_to_control_node_ratio']=paired['candidate_nodes']/paired['control_nodes']
    receipt=dict(status='complete',checked_at_utc=state['checked_at_utc'],kernel=state['kernel'],version=launch['version'],
        accelerator='none',seconds=result['seconds'],source_files_verified=source_count,
        payload_sha256=launch['payload_sha256'],result_sha256=sha(root/'result.json'),
        selected_arm=selection['selected_arm'],candidate=result['candidate'],control=control_summary,
        score_delta=result['score_delta'],calibration=selection['summaries'],solver=solver_stats,paired=paired,
        csv_sha256_reported=result['candidate_csv_sha256'],csv_downloaded_and_verified=False,
        aggregates_recomputed=True,official_metric_rerun_locally=False,leaderboard_submitted=False,
        decision='reject_E014_candidate_preserve_Harmonic_control',
        scope='Conditional development on 48 videos already used for design; not a leaderboard score')
    (ROOT/'results/E014_CPU_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
