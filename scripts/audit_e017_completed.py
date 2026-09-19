"""Verify completed E017 CPU and full-Harmonic inference receipts."""
import ast,base64,hashlib,io,json,math,zipfile
from pathlib import Path
from audit_cpu_control import aggregate
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(folder,notebook,launch,package):
    nb=read(ROOT/notebook/'notebook.ipynb');record=read(ROOT/launch)
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]));assert hashlib.sha256(payload).hexdigest()==record['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(folder/package/name).read_bytes(),name
        return len(archive.namelist())
def main():
    folder=ROOT/'outputs/e017_recovery';root=folder/'tissue_trajectory'
    cfolder=ROOT/'outputs/e017_control_recovery';control=cfolder/'calibration_control'
    count=verify(folder,'kaggle/tissue_trajectory','results/E017_CPU_launch.json','tissue_trajectory_package')
    ccount=verify(cfolder,'kaggle/calibration_control','results/E017_control_launch.json','calibration_control_package')
    result=read(root/'result.json');cr=read(control/'result.json')
    assert result['status']==cr['status']=='complete'
    assert read(folder/'status.json')['status']==read(cfolder/'status.json')['status']=='COMPLETE'
    names=read(ROOT/'baseline/event_graph_split.json')['split']['calibration']
    frozen=read(root/'frozen_predictions.json');assert frozen['videos']==names and not frozen['calibration_annotations_read']
    assert cr['videos']==names and not cr['annotations_read']
    reports=read(root/'trajectory_reports.json');assert set(reports)==set(names)
    for report in reports.values():
        assert not report['annotations_used']
        for v in report['starts'].values():assert v['objective_nondecreasing'] and v['final_objective']>=v['initial_objective']-1e-7
        best=max(report['starts'],key=lambda k:report['starts'][k]['final_objective'])
        assert report['selected_start']==best
    metric=read(root/'candidate_metrics.json');before=read(ROOT/'outputs/e016_recovery/identity_parent/geometric_metrics.json')
    assert metric['official_commit']==before['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
    assert sorted(r['dataset'] for r in metric['samples'])==sorted(names)
    for m,k in [(metric,'candidate'),(before,'control')]:
        assert all(math.isclose(v,m['summary'][key],abs_tol=1e-12,rel_tol=0) for key,v in aggregate(m['samples']).items())
        assert result[k]==m['summary']
    reference={r['dataset']:r for r in before['samples']}
    paired=[dict(dataset=r['dataset'],adj_edge_delta=r['adj_edge_jaccard']-reference[r['dataset']]['adj_edge_jaccard'],
        tp_delta=r['edge_tp']-reference[r['dataset']]['edge_tp'],fp_delta=r['edge_fp']-reference[r['dataset']]['edge_fp']) for r in metric['samples']]
    totals={label:{k:sum(r[k] for r in m['samples']) for k in ['edge_tp','edge_fp','edge_fn','num_pred_nodes']} for label,m in [('candidate',metric),('geometric',before)]}
    receipt=dict(status='complete',candidate=result['candidate'],geometric=result['control'],score_delta=result['score_delta'],
        seconds=result['seconds'],source_files_verified=count,control_source_files_verified=ccount,
        control_inference_seconds=cr['seconds'],control_csv_sha256_reported=cr['csv_sha256'],control_evaluation_pending=True,
        totals=totals,paired=paired,videos_improved=sum(r['adj_edge_delta']>0 for r in paired),videos_worse=sum(r['adj_edge_delta']<0 for r in paired),
        result_sha256=sha(root/'result.json'),control_result_sha256=sha(control/'result.json'),
        aggregates_recomputed=True,csv_downloaded_and_verified=False,official_metric_rerun_locally=False,leaderboard_submitted=False,
        decision='Retain trajectory hypothesis for full paired comparison; not yet evidence of beating Harmonic',scope='16 calibration videos; conditional development')
    (ROOT/'results/E017_completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
