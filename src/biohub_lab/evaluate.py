"""Score final, integer-coordinate CSVs using pinned official source."""
import json
import math
from pathlib import Path

from .submission import read_and_validate


def shapes_for(data_dir):
    import zarr
    return {p.stem: tuple(zarr.open_group(str(p), mode='r')['0'].shape)
            for p in sorted(Path(data_dir).glob('*.zarr'))}


def _find_estimate(value):
    if isinstance(value, dict):
        if 'estimated_number_of_nodes' in value:
            return float(value['estimated_number_of_nodes'])
        for child in value.values():
            result = _find_estimate(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_estimate(child)
            if result is not None:
                return result
    return None


def true_node_estimate(geff):
    for name in ('zarr.json', '.zattrs'):
        path = Path(geff) / name
        if path.exists():
            result = _find_estimate(json.loads(path.read_text()))
            if result is not None and math.isfinite(result) and result > 0:
                return result
    raise ValueError(f'Missing total-node estimate: {geff}; sparse GT count is NOT a substitute')


def evaluate_csv(csv_path, data_dir):
    import tracksdata as td
    import polars as pl
    from biohub_official.metrics import evaluate, node_recall, per_sample_metrics, summarise
    shapes = shapes_for(data_dir)
    datasets = read_and_validate(csv_path, shapes)
    rows = []
    for name, (nodes, edges) in datasets.items():
        graph = td.graph.InMemoryGraph()
        for key in ('z', 'y', 'x'):
            graph.add_node_attr_key(key, pl.Float64, 0.0)
        keys = sorted(nodes)
        ids = graph.bulk_add_nodes([{'t': nodes[k]['t'], **{axis: float(nodes[k][axis])
            for axis in ('z', 'y', 'x')}} for k in keys])
        mapping = dict(zip(keys, ids))
        if edges:
            graph.bulk_add_edges([{'source_id': mapping[s], 'target_id': mapping[t]} for s, t in edges])
        gt_path = Path(data_dir) / f'{name}.geff'
        truth = td.graph.IndexedRXGraph.from_geff(str(gt_path))[0]
        result = evaluate(graph, truth, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
        recall = node_recall(graph, truth) if graph.num_edges() else float('nan')
        row = per_sample_metrics(result, true_node_estimate(gt_path), recall)
        row['dataset'] = name
        rows.append(row)
    summary = summarise(rows)
    if summary['n'] != len(shapes) or summary['n_adj'] != len(shapes):
        raise ValueError('Incomplete metric coverage; refuse to silently skip samples')
    return {'summary': summary, 'samples': rows,
            'scope': 'training diagnostic; pretrained checkpoint training membership unverified',
            'official_commit': '075fc5f5a52d11077f9dc2b074644618f26939e2'}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate_csv(args.csv, args.data_dir)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
