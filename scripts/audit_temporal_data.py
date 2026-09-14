"""Inventory annotations and pretrained split provenance without reading test images."""
import json
from pathlib import Path
import numpy as np
import zarr


def main():
    candidates = [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
                  Path('/kaggle/input/biohub-cell-tracking-during-development')]
    comp = next(p for p in candidates if (p/'train').exists())
    rows = []
    for p in sorted((comp/'train').glob('*.geff')):
        g = zarr.open_group(str(p), mode='r')
        ids = np.asarray(g['nodes/ids'][:])
        coords = np.column_stack([g[f'nodes/props/{k}/values'][:] for k in ('t','z','y','x')])
        edge_ids = np.asarray(g['edges/ids'][:])
        if edge_ids.ndim != 2 or edge_ids.shape[1] != 2:
            raise ValueError(f'Unexpected GEFF edge shape {edge_ids.shape}')
        sources, counts = np.unique(edge_ids[:,0], return_counts=True)
        image = zarr.open_group(str(p.with_suffix('.zarr')),mode='r')
        shape = image['0'].shape
        rows.append(dict(name=p.stem,group=p.stem.split('_')[0],shape=list(shape),nodes=len(ids),
            edges=len(edge_ids),divisions=int((counts==2).sum()),
            annotated_frames=len(np.unique(coords[:,0])),
            coord_min=coords.min(0).tolist(),coord_max=coords.max(0).tolist(),
            graph_attrs=dict(g.attrs),image_attrs=dict(image.attrs)))
        print(p.stem,'nodes',len(ids),'edges',len(edge_ids),'divisions',rows[-1]['divisions'],flush=True)
    manifests=[]
    asset_roots=list(Path('/kaggle/input/datasets/pilkwang').glob('*'))
    asset_roots += [p for p in Path('/kaggle/input').glob('biohub-*') if p != comp]
    for name in ('split_manifest.json','training_config.json','gate_summary.json'):
        for root in asset_roots:
            for path in root.rglob(name):
                manifests.append(dict(path=str(path),content=json.loads(path.read_text())))
    out=dict(videos=rows,pretrained_manifests=manifests,
        test_names=sorted(p.stem for p in (comp/'test').glob('*.zarr')),
        scope='Metadata and sparse annotations only; no test labels used')
    Path('/kaggle/working/temporal_data_audit.json').write_text(json.dumps(out,indent=2))
    print('AUDIT_COMPLETE',len(rows),'videos',sum(x['divisions'] for x in rows),'divisions',flush=True)


if __name__=='__main__': main()
