"""Audit E016 source provenance, fixed training and paired calibration reports."""
import ast,base64,hashlib,io,json,math,zipfile
from pathlib import Path
from audit_cpu_control import aggregate

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    folder=ROOT/'outputs/e016_recovery';root=folder/'identity_parent'
    result=read(root/'result.json');state=read(folder/'status.json');launch=read(ROOT/'results/E016_launch.json')
    assert state['status']=='COMPLETE' and result['status']=='complete'
    assert result['accelerator']=='none' and not launch['enable_gpu'] and not launch['enable_tpu']
    assert not result['evaluation_cohort_used'] and not result['leaderboard_submitted']
    nb=read(ROOT/'kaggle/identity_parent/notebook.ipynb')
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]));assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(folder/'identity_parent_package'/name).read_bytes(),name
        source_count=len(archive.namelist())
    config=read(root/'config.json');split=read(ROOT/'baseline/event_graph_split.json')['split']
    assert config['fit']==split['fit'] and config['calibration']==split['calibration']
    assert config['config']['steps']==result['steps']==3000
    training=read(root/'training.json');assert not training['calibration_annotations_read']
    assert [r['step'] for r in training['history']]==list(range(250,3001,250))
    assert all(math.isfinite(r['loss']) for r in training['history'])
    counts=read(root/'fit_counts.json');assert set(counts)==set(split['fit'])
    frozen=read(root/'frozen_predictions.json');assert not frozen['calibration_annotations_read']
    assert frozen['checkpoint_sha256']==result['checkpoint_sha256']
    rows={};totals={}
    for arm in ['learned','geometric']:
        metric=read(root/(arm+'_metrics.json'))
        assert metric['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
        assert sorted(r['dataset'] for r in metric['samples'])==sorted(split['calibration'])
        assert all(math.isfinite(v) for r in metric['samples'] for v in r.values() if isinstance(v,(int,float)))
        computed=aggregate(metric['samples'])
        assert all(math.isclose(v,metric['summary'][k],abs_tol=1e-12,rel_tol=0) for k,v in computed.items())
        assert metric['summary']==result['summaries'][arm]
        rows[arm]={r['dataset']:r for r in metric['samples']}
        totals[arm]={k:sum(r[k] for r in metric['samples']) for k in ['edge_tp','edge_fp','edge_fn','num_pred_nodes']}
    delta=result['summaries']['learned']['score']-result['summaries']['geometric']['score']
    assert math.isclose(delta,result['score_delta'],abs_tol=1e-12)
    paired=[]
    for name in split['calibration']:
        a,b=rows['learned'][name],rows['geometric'][name]
        paired.append(dict(dataset=name,adjusted_edge_delta=a['adj_edge_jaccard']-b['adj_edge_jaccard'],
            raw_edge_delta=a['edge_jaccard']-b['edge_jaccard'],tp_delta=a['edge_tp']-b['edge_tp'],
            fp_delta=a['edge_fp']-b['edge_fp'],fn_delta=a['edge_fn']-b['edge_fn']))
    receipt=dict(status='complete',kernel=state['kernel'],version=launch['version'],checked_at_utc=state['checked_at_utc'],
        result_sha256=sha(root/'result.json'),source_files_verified=source_count,training_steps=3000,seconds=result['seconds'],
        accelerator='none',fit_videos=48,calibration_videos=16,summaries=result['summaries'],score_delta=delta,
        totals=totals,paired=paired,videos_improved=sum(r['adjusted_edge_delta']>0 for r in paired),
        videos_worse=sum(r['adjusted_edge_delta']<0 for r in paired),videos_tied=sum(r['adjusted_edge_delta']==0 for r in paired),
        fit_examples=sum(r['examples'] for r in counts.values()),null_fit_examples=sum(r['null_examples'] for r in counts.values()),
        csv_sha256_reported=frozen['csv_sha256'],checkpoint_sha256_reported=frozen['checkpoint_sha256'],
        aggregates_recomputed=True,official_metric_rerun_locally=False,csv_downloaded_and_verified=False,
        checkpoint_downloaded_and_verified=False,leaderboard_submitted=False,
        decision='Do not submit or extend training: small paired gain; next compare full Harmonic on same calibration cohort and diagnose disagreement graph edits.',
        scope='Conditional calibration; geometric control is not full Harmonic; E015-to-E016 changes confound detection set and training')
    (ROOT/'results/E016_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
