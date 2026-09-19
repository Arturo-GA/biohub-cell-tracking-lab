"""Explicit upper-border rounding repair, preserving original CSV and topology."""
import csv
from pathlib import Path
from .submission import COLUMNS,read_and_validate

def export_control(source,target,shapes):
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():raise ValueError('Preserve original')
    changes={};nodes=edges=0
    with source.open(newline='') as inp,target.open('w',newline='') as out:
        reader=csv.DictReader(inp);writer=csv.DictWriter(out,fieldnames=COLUMNS)
        if reader.fieldnames!=COLUMNS:raise ValueError('Unexpected schema')
        writer.writeheader()
        for row in reader:
            if row['row_type']=='node':
                nodes+=1
                for axis,extent in zip(('t','z','y','x'),shapes[row['dataset']]):
                    value=float(row[axis])
                    if not value.is_integer():raise ValueError('Noninteger export')
                    if axis!='t' and value==extent:
                        row[axis]=str(extent-1);changes[axis]=changes.get(axis,0)+1
                    elif not 0<=value<extent:raise ValueError('Unexpected coordinate excursion')
            elif row['row_type']=='edge':edges+=1
            writer.writerow(row)
    read_and_validate(target,shapes)
    return dict(changed_spatial_coordinates=changes,nodes=nodes,edges=edges,topology_changed=False,
        rule='Only spatial integer equal to upper extent maps to extent-1; all other excursions rejected')
