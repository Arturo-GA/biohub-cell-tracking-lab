"""Audit downloaded E047 metrics and close the E045-E048 experiment batch."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def main():
    receipt = json.loads((ROOT / 'results/E047_RECOVER_completed.json').read_text())
    result = receipt['result']
    audit = {}
    for arm, summary in result['metrics'].items():
        metric = json.loads((ROOT / f'outputs/e047_recover/consensus_graph/{arm}_metrics.json').read_text())
        counts = {key: sum(sample[key] for sample in metric['samples']) for key in ('edge_tp', 'edge_fp', 'edge_fn')}
        assert all(counts[key] == summary[key] for key in counts)
        audit[arm] = {'counts': counts, 'delta_score': summary['score'] - result['metrics']['control']['score'],
                      'delta_by_embryo': {e: m['score'] - result['embryo_metrics']['control'][e]['score'] for e, m in result['embryo_metrics'][arm].items()}}
    assert not any(result['passed'].values())
    assert result['csv_sha256']['control'] == '9d39a33fa0f4132619b91bb41ff261df428f95126f82a1f2678b963ba1a92ef4'
    write('results/E047_completed.json', receipt)
    write('results/E047_counts_audit.json', audit)
    state = json.loads((ROOT / 'results/STATUS.json').read_text())
    state['E047'].update(status='COMPLETE', seconds=result['seconds'], passed=result['passed'],
                         receipt='results/E047_completed.json', decision='No candidate passed; extra false links and embryo regression')
    state.update(active_experiment=None, active_experiments=[], next_experiment_started=False,
                 quality_status='E045-E048 complete. No candidate passed its gate; no new submission.',
                 next_research_direction='Public DAE adaptation and real dense temporal supervision are unvalidated research possibilities; do not repeat threshold sweeps on reused calibration.')
    write('results/STATUS.json', state)
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
