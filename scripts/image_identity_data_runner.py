"""CPU E030 annotated/jittered nucleus training curriculum, not detector validation."""
import json, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
import zarr, tracksdata as td
from biohub_lab.image_identity import physical_views, candidate_events, SCALE


def main(package):
    start = time.monotonic(); root = Path('/kaggle/working/image_identity_data'); root.mkdir(exist_ok=True)
    config = json.loads((package/'baseline/e030_image_identity.json').read_text())
    inputs = Path('/kaggle/input')
    train = next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development', inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    assert not set(config['fit']) & set(config['validation'])
    records = []; rng = np.random.default_rng(config['seed']); keys = td.DEFAULT_ATTR_KEYS
    for split in ('fit', 'validation'):
        for video in config[split]:
            image = zarr.open_group(str(train/(video+'.zarr')), mode='r')['0']
            graph = td.graph.IndexedRXGraph.from_geff(train/(video+'.geff')); graph = graph[0] if isinstance(graph, tuple) else graph
            rows = list(graph.node_attrs().iter_rows(named=True)); rows.sort(key=lambda r:(r['t'],r[keys.NODE_ID]))
            index = {r[keys.NODE_ID]:i for i,r in enumerate(rows)}
            gt = np.array([[r[a] for a in ('t','z','y','x')] for r in rows], float)
            ge = list(graph.edge_attrs(attr_keys=[keys.EDGE_SOURCE, keys.EDGE_TARGET]).iter_rows(named=True))
            parents = {index[r[keys.EDGE_TARGET]]:index[r[keys.EDGE_SOURCE]] for r in ge}
            counts = Counter(parents.values())
            available = sorted({int(gt[a,0]) for b,a in parents.items() if gt[b,0] == gt[a,0]+1})
            if not available: continue
            chosen = set(np.asarray(available)[np.linspace(0,len(available)-1,min(config['transitions_per_video'],len(available)),dtype=int)].tolist())
            if split == 'fit': chosen.update(int(gt[a,0]) for a,c in counts.items() if c == 2)
            frames = chosen | {t+1 for t in chosen}
            selected = np.flatnonzero(np.isin(gt[:,0], sorted(frames)))
            # Two independent localization perturbations imitate cross-detector centers.
            # Both jittered views exist at fit AND validation; no exact-center shortcut.
            coords = np.repeat(gt[selected], 2, axis=0); identity = np.repeat(selected, 2)
            coords[:,1:] += rng.uniform(-config['jitter_um'],config['jitter_um'],(len(coords),3))/SCALE
            coords[:,1:] = np.clip(coords[:,1:],0,np.array(image.shape[1:])-1)
            crops = np.empty((len(coords),3,64,64),np.float16)
            for t in sorted(frames):
                volume = np.asarray(image[t],np.float32); low,high = np.percentile(volume,[2,99.8])
                volume = np.clip((volume-low)/max(float(high-low),1e-8),0,1)
                for i in np.flatnonzero(coords[:,0] == t): crops[i] = physical_views(volume,coords[i,1:])
            # Training one representative per identity avoids teaching duplicate parents.
            canonical = np.arange(0,len(coords),2); canonical_coords = coords[canonical]
            edge_local, triple_local = candidate_events(canonical_coords,config['candidate_k'],config['radius_um'])
            edges = canonical[edge_local]; triples = canonical[triple_local]
            ep = np.array([parents.get(int(identity[b]),-1) for a,b in edges])
            known = (ep>=0) & np.isin(coords[edges[:,0],0],sorted(chosen))
            edges, ep = edges[known], ep[known]; edge_y = (identity[edges[:,0]] == ep).astype(np.float32)
            ty, tk = [], []
            for m,a,b in triples:
                pa,pb = parents.get(int(identity[a]),-1),parents.get(int(identity[b]),-1)
                positive = pa == identity[m] and pb == identity[m] and identity[a] != identity[b]
                negative = (pa>=0 and pa!=identity[m]) or (pb>=0 and pb!=identity[m])
                ty.append(positive); tk.append((positive or negative) and coords[m,0] in chosen)
            triples = triples[np.asarray(tk,bool)]; triple_y = np.asarray(ty,np.float32)[np.asarray(tk,bool)]
            pairs = [(i,i+1) for i in canonical]; pair_y = [1.]*len(pairs)
            for i in canonical:
                others = canonical[(coords[canonical,0] == coords[i,0]) & (identity[canonical] != identity[i])]
                order = np.argsort(np.linalg.norm((coords[others,1:]-coords[i,1:])*SCALE,axis=1),kind='stable')[:2]
                for j in others[order]: pairs.append((i,int(j))); pair_y.append(0.)
            path = root/(video+'.npz')
            np.savez_compressed(path,crops=crops,coords=coords.astype(np.float32),identity=identity,edges=edges,edge_y=edge_y,triples=triples,triple_y=triple_y,pairs=np.asarray(pairs,np.int64).reshape(-1,2),pair_y=np.asarray(pair_y,np.float32))
            record = dict(split=split,video=video,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nodes=len(coords),edges=len(edges),edge_positive=int(edge_y.sum()),triples=len(triples),division_positive=int(triple_y.sum()),frames=sorted(frames))
            records.append(record); print('IDENTITY_DATA',json.dumps(record),flush=True)
    result = dict(status='complete',seconds=time.monotonic()-start,config=config,videos=records,scope='Annotation-centered jitter curriculum; not detector-space or official graph validation')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('IDENTITY_DATA_READY',len(records),flush=True)

if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))
