"""Audit recovered E013 provenance and paired metric reports, without rerunning GT."""
import ast
import base64
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from audit_cpu_control import aggregate, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    recovered = ROOT/'outputs/e013_candidate_recovery'
    root = recovered/'event_graph_experiment'
    read = lambda path: json.loads(path.read_text())
    result = read(root/'result.json')
    train = read(root/'training.json')
    launch = read(ROOT/'results/E013_launch.json')
    split = read(ROOT/'baseline/event_graph_split.json')['split']
    assert result['status'] == train['status'] == 'complete'
    assert result['split'] == split
    assert train['config']['steps'] == train['history'][-1]['step'] == 6000
    package = recovered/'event_graph_package'
    assert read(package/'baseline/event_graph_split.json') == read(ROOT/'baseline/event_graph_split.json')
    assert read(package/'baseline/event_graph_inputs.json') == read(ROOT/'baseline/event_graph_inputs.json')
    assert result['split_manifest_sha256'] == sha(package/'baseline/event_graph_split.json')
    assert result['e012_input_pins_sha256'] == sha(package/'baseline/event_graph_inputs.json')
    assert result['frozen_inputs_sha256'] == sha(root/'frozen_inputs.json')
    frozen = read(root/'frozen_inputs.json')
    expected = set(split['fit'] + split['calibration'] + split['evaluation'])
    assert set(frozen['records']) == expected and len(expected) == 112
    assert sha(root/'best.pt') == result['calibration']['checkpoint_sha256']
    predictions = read(root/'frozen_predictions.json')
    assert predictions['checkpoint_sha256'] == sha(root/'best.pt')
    assert predictions['calibration_sha256'] == sha(root/'calibration.json')
    notebook = read(ROOT/'kaggle/event_graph/notebook.ipynb')
    payload = None
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'payload' for t in node.targets):
                payload = base64.b64decode(ast.literal_eval(node.value.args[0]))
    import hashlib
    assert hashlib.sha256(payload).hexdigest() == launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            assert (recovered/'event_graph_package'/name).read_bytes() == archive.read(name), name
        source_count = len(archive.namelist())
    for name, record in frozen['records'].items():
        assert record == read(root/'videos'/name/'generation.json')
        assert record['runtime_sha256'] == sha(root/'runtime.json')
        assert record['annotations_read'] is False
    candidate = read(root/'candidate_metrics.json')
    assert candidate == result['candidate']
    control = read(ROOT/'outputs/cpu_finished_check/control/control_cpu/control_metrics.json')
    summaries = []
    for metrics in (candidate, control):
        assert metrics['official_commit'] == '075fc5f5a52d11077f9dc2b074644618f26939e2'
        assert sorted(r['dataset'] for r in metrics['samples']) == sorted(split['evaluation'])
        assert all(math.isfinite(v) for r in metrics['samples'] for v in r.values() if isinstance(v, (int, float)))
        summary = aggregate(metrics['samples'])
        assert all(math.isclose(v, metrics['summary'][k], abs_tol=1e-12, rel_tol=0) for k,v in summary.items())
        summaries.append(summary)
    reports = [read(root/'videos'/name/'prediction.json')['solver'] for name in split['evaluation']]
    receipt = dict(status='complete', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        kernel='jarturo/biohub-lab-learned-event-graph', version=1,
        neural_training=True, training_steps=6000, selected_step=train['selected_step'],
        candidate=summaries[0], control=summaries[1],
        score_delta=summaries[0]['score']-summaries[1]['score'],
        source_files_verified=source_count, payload_sha256=launch['payload_sha256'],
        checkpoint_sha256=sha(root/'best.pt'), last_checkpoint_sha256=sha(root/'last.pt'),
        result_sha256=sha(root/'result.json'), frozen_inputs_sha256=sha(root/'frozen_inputs.json'),
        feature_video_receipts_verified=112, graph_feature_arrays_downloaded=False,
        solver_windows=sum(len(r['windows']) for r in reports),
        solver_fallback_windows=sum(r['fallback_windows'] for r in reports),
        metrics_aggregation_recomputed=True, official_metric_rerun_locally=False,
        csv_recovered=False, leaderboard_submitted=False, decision='reject_current_candidate',
        scope=result['scope'])
    (ROOT/'results/E013_completed_comparison.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
