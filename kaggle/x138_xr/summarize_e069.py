"""Compare completed E069 labs with their matching controls; never submit."""
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/E069'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score(rows, stems):
    union = sum(rows[s]['edge_tp'] + rows[s]['edge_fp'] + rows[s]['edge_fn'] for s in stems)
    adjusted = sum(rows[s]['adj_edge_jaccard'] *
                   (rows[s]['edge_tp'] + rows[s]['edge_fp'] + rows[s]['edge_fn']) for s in stems)
    div_tp = sum(rows[s]['division_tp'] for s in stems)
    div_union = sum(rows[s]['division_tp'] + rows[s]['division_fp'] + rows[s]['division_fn'] for s in stems)
    return adjusted / max(union, 1) + .1 * div_tp / max(div_union, 1)


def compare(rows, control, stems):
    deltas = {s: score(rows, [s]) - score(control, [s]) for s in stems}
    return dict(delta=score(rows, stems) - score(control, stems),
                wins=sum(v > 1e-12 for v in deltas.values()),
                losses=sum(v < -1e-12 for v in deltas.values()),
                changed_metric_videos=[s for s in stems if rows[s] != control[s]],
                video_deltas={s: v for s, v in deltas.items() if abs(v) > 1e-12})


def summarize(stage, mode, expected_videos, expected_configurations):
    receipt = read(RESULT / (stage + '_completed.json'))
    assert receipt['manifest_verified'] is True
    folder = ROOT / 'outputs/e069' / stage / ('e069_' + mode)
    complete = read(folder / 'complete.json')
    assert complete['videos'] == expected_videos
    assert complete['configurations'] == expected_configurations
    results = read(folder / 'results.json')
    assert len(results) == expected_configurations
    rows = {r['name']: read(folder / (r['name'] + '.rows.json')) for r in results}
    stems = sorted(rows['C3_public0955'])
    assert len(stems) == expected_videos
    assert all(set(v) == set(stems) for v in rows.values())
    panel = read(folder / 'panel.json') if (folder / 'panel.json').exists() else []
    groups = {g: [s for s in stems if s.startswith(g)] for g in ('44b6', '6bba')}
    groups.update({subset: [r['stem'] for r in panel if r['subset'] == subset]
                   for subset in ('development', 'confirmation')} if panel else {})
    comparisons = []
    for result in results:
        name = result['name']
        actual = score(rows[name], stems)
        assert abs(actual - result['score']) < 1e-10, (name, actual, result['score'])
        control_name = 'C3_fork8' if name.startswith('fork8_') else 'C3_public0955'
        comparison = compare(rows[name], rows[control_name], stems)
        comparison.update(name=name, score=actual, control=control_name,
                          groups={g: compare(rows[name], rows[control_name], ss)
                                  for g, ss in groups.items()},
                          operations=result['operations'], division=result['division'])
        if name.startswith('neural_'):
            comparison['versus_geometry'] = compare(rows[name], rows['geometry'], stems)
        comparison['has_positive_increment'] = comparison['delta'] > 1e-12
        comparisons.append(comparison)
    report = dict(kernel=receipt['kernel'], version=receipt['version'],
                  videos=expected_videos, configurations=expected_configurations,
                  comparisons=comparisons)
    (RESULT / (stage + '_comparisons.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    reports = [summarize('duplicates', 'duplicates', 199, 6),
               summarize('local_eval', 'local_linker_eval', 32, 11)]
    lines = []
    for report in reports:
        lines.append(f"\n{report['kernel']} ({report['videos']} videos)")
        for row in report['comparisons']:
            lines.append(f"{row['name']:27} score={row['score']:.9f} "
                         f"increment={row['delta']:+.9f} vs {row['control']} "
                         f"wins/losses={row['wins']}/{row['losses']}")
    note = ('Training/tuning diagnostics. The 32-video panel and the 199-video lab are not directly comparable. '
            'An increment over C3_fork8 is required to credit its combination to a new rule. '
            'Positive panel changes require broader validation before promotion; no prediction of public score.')
    (RESULT / 'comparison_summary.json').write_text(json.dumps(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), note=note, labs=reports), indent=2) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(note)


if __name__ == '__main__':
    main()
