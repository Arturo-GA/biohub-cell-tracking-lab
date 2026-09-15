"""Verify a recovered CPU control run against its frozen cohort and export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate(rows):
    """Independently recompute the pinned official summary from its sample rows."""
    totals = {key: sum(row[key] for row in rows) for key in
              ('edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn')}
    edge_total = sum(totals[key] for key in ('edge_tp', 'edge_fp', 'edge_fn'))
    div_total = sum(totals[key] for key in ('division_tp', 'division_fp', 'division_fn'))
    adjusted = sum(sum(row[key] for key in ('edge_tp', 'edge_fp', 'edge_fn')) *
                   row['adj_edge_jaccard'] for row in rows) / edge_total
    division = totals['division_tp'] / div_total
    return dict(n=len(rows), n_adj=len(rows), edge_jaccard=totals['edge_tp']/edge_total,
                division_jaccard=division, division_tp=totals['division_tp'],
                division_fp=totals['division_fp'], division_fn=totals['division_fn'],
                node_recall=sum(row['node_recall'] for row in rows)/len(rows),
                adj_edge_jaccard=adjusted, score=adjusted + .1 * division)


def audit(folder):
    folder = Path(folder)
    download = json.loads((folder/'download_verification.json').read_text())
    launch = json.loads((ROOT/'results/E013_cpu_control_launch.json').read_text())
    state = json.loads((folder/'status.json').read_text())
    metrics_path = folder/'control_cpu/control_metrics.json'
    metrics = json.loads(metrics_path.read_text())
    result = json.loads((folder/'control_cpu/result.json').read_text())
    repair = json.loads((folder/'control_cpu/control_cpu_v2.repair.json').read_text())
    original_repair = json.loads((ROOT/'results/E013_control_export_repair.json').read_text())
    expected = json.loads((ROOT/'baseline/event_graph_split.json').read_text())['split']['evaluation']
    assert state['status'] == 'COMPLETE' and result['status'] == 'complete'
    assert download['payload_sha256'] == launch['payload_sha256']
    assert result['control'] == metrics
    assert len(metrics['samples']) == 48
    assert sorted(row['dataset'] for row in metrics['samples']) == sorted(expected)
    assert metrics['official_commit'] == '075fc5f5a52d11077f9dc2b074644618f26939e2'
    for row in metrics['samples']:
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
    summary = aggregate(metrics['samples'])
    assert summary.keys() == metrics['summary'].keys()
    assert all(math.isclose(value, metrics['summary'][key], rel_tol=0, abs_tol=1e-12)
               for key, value in summary.items())
    csv_sha = sha(folder/'control_cpu/control_cpu_v2.csv')
    assert csv_sha == result['csv_sha256'] == repair['corrected_sha256'] == original_repair['corrected_sha256']
    assert all(value == original_repair[key] for key, value in repair.items())
    assert sum(row['num_pred_nodes'] for row in metrics['samples']) == repair['nodes']
    receipt = dict(kernel=state['kernel'], version=launch['version'], status='COMPLETE',
        checked_at_utc=state['checked_at_utc'], verified_at_utc=datetime.now(timezone.utc).isoformat(),
        summary=summary, seconds=result['seconds'], payload_sha256=download['payload_sha256'],
        csv_sha256=csv_sha, metrics_sha256=sha(metrics_path), official_commit=metrics['official_commit'],
        groups={group:aggregate([r for r in metrics['samples'] if r['dataset'].startswith(group+'_')])
                for group in ('44b6','6bba')},
        verification=dict(remote_packaged_sources_match_launch_payload=True,
            downloaded_csv_matches_validated_local_export=True, exact_48_video_cohort=True,
            all_sample_metrics_finite=True, aggregate_recomputed=True, node_count_preserved=True,
            export_repair_matches_original_receipt=True),
        nodes=repair['nodes'], edges=repair['edges'], topology_changed=False,
        scope='Conditional development control on the fixed E012 cohort; not an independent validation or leaderboard score.',
        gpu=False, tpu=False, leaderboard_submitted=False,
        decision='Use as the paired reference for the learned event graph; no candidate improvement demonstrated yet.')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='outputs/cpu_finished_check/control')
    parser.add_argument('--output', default='results/E013_cpu_control_completed.json')
    args=parser.parse_args(); result=audit(args.folder)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
