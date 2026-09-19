"""Verify E015 provenance and recompute aggregate metrics from downloaded reports."""
import ast,base64,hashlib,io,json,math,zipfile
from pathlib import Path
from audit_cpu_control import aggregate

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    folder=ROOT/'outputs/e015_recovery';root=folder/'detection_identity'
    state=read(folder/'status.json');result=read(root/'result.json')
    launch=read(ROOT/'results/E015_launch.json')
    assert state['status']=='COMPLETE' and result['status']=='complete'
    assert result['accelerator']=='none' and not launch['enable_gpu'] and not result['evaluation_cohort_used']
    nb=read(ROOT/'kaggle/detection_identity/notebook.ipynb')
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(folder/'detection_identity_package'/name).read_bytes(),name
        source_count=len(archive.namelist())
    pins=read(ROOT/'baseline/e015_identity_pins.json')
    before_path=ROOT/'outputs/e014_cpu_recovery/association_cpu/calibration_continuations_metrics.json'
    assert sha(before_path)==pins['baseline_metrics_sha256']
    before=read(before_path);after=read(root/'projected_metrics.json')
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    frozen=read(root/'frozen_prediction.json');audit=read(root/'identity_audit.json')
    assert frozen['videos']==names and not frozen['annotations_read'] and set(audit)==set(names)
    for label,metric in [('original',before),('projected',after)]:
        assert metric['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        assert sorted(r['dataset'] for r in metric['samples'])==sorted(names)
        summary=aggregate(metric['samples'])
        assert all(math.isclose(v,metric['summary'][k],rel_tol=0,abs_tol=1e-12) for k,v in summary.items())
        assert metric['summary']==result[label]
    reference={r['dataset']:r for r in before['samples']}
    paired={key:sum(r[key]>reference[r['dataset']][key] for r in after['samples']) for key in ['edge_jaccard','adj_edge_jaccard']}
    totals={}
    for arm in ['original','projected']:
        values=[r[arm] for r in audit.values()]
        totals[arm]={k:sum(v[k] for v in values) for k in ['predicted_nodes','annotated_nodes','many_to_one_matched_predictions','many_to_one_covered_gt','gt_with_multiple_predictions','one_to_one_matched_predictions']}
        for mapping in ['many_to_one_edges','one_to_one_edges']:
            totals[arm][mapping]={k:sum(v[mapping][k] for v in values) for k in values[0][mapping]}
        assert totals[arm]['predicted_nodes']==sum(r['num_pred_nodes'] for r in (before if arm=='original' else after)['samples'])
    receipt=dict(kernel=state['kernel'],version=launch['version'],status='complete',checked_at_utc=state['checked_at_utc'],
        result_sha256=sha(root/'result.json'),source_files_verified=source_count,seconds=result['seconds'],accelerator='none',
        original=result['original'],projected=result['projected'],score_delta=result['calibration_score_delta'],
        videos_improved=paired,identity_diagnostics=totals,aggregates_recomputed=True,
        official_metric_rerun_locally=False,csv_downloaded_and_verified=False,csv_sha256_reported=frozen['csv_sha256'],
        leaderboard_submitted=False,decision='Identity discrepancy supported; projection alone is insufficient. Next: identity-aware supervision and competitive parent assignment.',
        scope='16 calibration videos; no paired Harmonic score for this cohort; not a leaderboard improvement')
    (ROOT/'results/E015_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
