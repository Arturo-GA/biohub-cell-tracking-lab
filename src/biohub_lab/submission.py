"""Validate exactly the final CSV; no silent repair or dropped datasets."""
import csv
import math
from collections import Counter
from pathlib import Path

COLUMNS = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']


def read_and_validate(path, shapes):
    """Return dataset -> (node dictionaries, edge tuples), checking physical bounds."""
    groups = {}
    with Path(path).open(newline='', encoding='utf8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError('Incorrect submission columns')
        for index, row in enumerate(reader):
            ints = {}
            for col in COLUMNS:
                if col in ('dataset', 'row_type'):
                    continue
                value = float(row[col])
                if not math.isfinite(value) or value != int(value):
                    raise ValueError(f'Non-integer value at row {index}, {col}')
                ints[col] = int(value)
            if ints['id'] != index:
                raise ValueError('Row IDs must be contiguous from zero')
            dataset = row['dataset']
            if dataset not in shapes:
                raise ValueError(f'Unexpected dataset: {dataset}')
            nodes, edges = groups.setdefault(dataset, ({}, []))
            if row['row_type'] == 'node':
                node_id = ints['node_id']
                if node_id < 0 or node_id in nodes:
                    raise ValueError(f'{dataset}: invalid/duplicate node ID')
                for col, size in zip(('t', 'z', 'y', 'x'), shapes[dataset]):
                    if not 0 <= ints[col] < size:
                        raise ValueError(f'{dataset}: {col} out of bounds')
                if ints['source_id'] != -1 or ints['target_id'] != -1:
                    raise ValueError('Node edge sentinels must be -1')
                nodes[node_id] = {k: ints[k] for k in ('t', 'z', 'y', 'x')}
            elif row['row_type'] == 'edge':
                if any(ints[k] != -1 for k in ('node_id', 't', 'z', 'y', 'x')):
                    raise ValueError('Edge node sentinels must be -1')
                edges.append((ints['source_id'], ints['target_id']))
            else:
                raise ValueError('Unknown row_type')
    if set(groups) != set(shapes):
        raise ValueError(f'Missing datasets: {sorted(set(shapes)-set(groups))}')
    for dataset, (nodes, edges) in groups.items():
        if not nodes:
            raise ValueError(f'{dataset}: empty node set')
        if len(edges) != len(set(edges)):
            raise ValueError(f'{dataset}: duplicate edge')
        incoming, outgoing = Counter(), Counter()
        for source, target in edges:
            if source not in nodes or target not in nodes:
                raise ValueError(f'{dataset}: dangling edge')
            if nodes[target]['t'] != nodes[source]['t'] + 1:
                raise ValueError(f'{dataset}: non-consecutive edge')
            incoming[target] += 1
            outgoing[source] += 1
        if max(incoming.values(), default=0) > 1 or max(outgoing.values(), default=0) > 2:
            raise ValueError(f'{dataset}: invalid lineage degree')
    return groups
