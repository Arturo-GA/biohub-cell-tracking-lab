"""Explicit versioned correction of the recovered E013 integer export."""
import csv
from pathlib import Path

from .event_portable import sha
from .event_data import save_json
from .submission import COLUMNS, read_and_validate

# Kept separately from the historical notebook: verify the actual recovered file.
E013_CONTROL_CSV_SHA256 = '2c714da977670aa753394ab56928ace6efbb40fe1ae0a74c9cb94671a3d9b4b4'


def repair_export(source, destination, shapes, expected_sha=None, expected_changes=1):
    """Project only spatial integers equal to the upper extent onto extent-1.

    This is the nearest valid voxel to a rounded upper-border point. Larger
    excursions, negative values, temporal faults and nonintegers are rejected.
    Original predictions and topology are preserved in the source artifact.
    """
    source = Path(source); destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Preserve the original CSV; choose another destination')
    source_hash = sha(source)
    if expected_sha is not None and source_hash != expected_sha:
        raise ValueError('Recovered control CSV checksum mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True); temp = destination.with_suffix('.csv.tmp')
    changes = {}; nodes = edges = 0
    with source.open(newline='') as inp, temp.open('w', newline='') as out:
        reader = csv.DictReader(inp); writer = csv.DictWriter(out, fieldnames=COLUMNS)
        if reader.fieldnames != COLUMNS:
            raise ValueError('Unexpected CSV schema')
        writer.writeheader()
        for row in reader:
            if row['row_type'] == 'node':
                nodes += 1; name = row['dataset']
                for axis, extent in zip(('t', 'z', 'y', 'x'), shapes[name]):
                    value = int(row[axis])
                    if axis != 't' and value == extent:
                        row[axis] = str(extent-1); changes[axis] = changes.get(axis, 0)+1
                    elif not 0 <= value < extent:
                        raise ValueError('Unexpected coordinate excursion; refuse broad clipping')
            elif row['row_type'] == 'edge':
                edges += 1
            writer.writerow(row)
    if sum(changes.values()) != expected_changes:
        raise ValueError('Unexpected number of coordinate corrections')
    read_and_validate(temp, shapes)
    temp.replace(destination)
    report = dict(version='E013-control-cpu-export-v2', source_sha256=source_hash, corrected_sha256=sha(destination),
        changed_spatial_coordinates=changes, nodes=nodes, edges=edges, topology_changed=False,
        rule='Nearest valid integer voxel for upper-border rounding; reject all other out-of-range values',
        historical_csv_preserved=True, official_metric_computed=False)
    save_json(destination.with_suffix('.repair.json'), report)
    return report
