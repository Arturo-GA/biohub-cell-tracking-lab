"""Run morphology/HOCT on fixed detections and score the final integer CSV."""
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import zarr

from biohub_lab.hoct_linker import HOCTConfig, candidate_edges, frame_features, load_hoct, score_edges, solve_lineage
from biohub_lab.submission import COLUMNS, read_and_validate
from biohub_lab.evaluate import evaluate_csv, shapes_for


def input_assets():
    matches = list(Path('/kaggle/input').rglob('hoct_model_manifest.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one attached HOCT asset dataset, found {matches}')
    return matches[0].parent


def write_csv(path, results):
    index = 0
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for name, (coords, edges) in sorted(results.items()):
            for n, (t,z,y,x) in enumerate(coords):
                writer.writerow([index,name,'node',n,int(t),int(z),int(y),int(x),-1,-1])
                index += 1
            for s,t in edges:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(s),int(t)])
                index += 1


def main(mode, package):
    sys.path.insert(0, str(Path(package)/'scripts'))
    from notebook_runner import competition_dir
    comp = competition_dir()
    assets = input_assets()
    manifest = json.loads((assets/'hoct_model_manifest.json').read_text())
    root = Path('/kaggle/working/hoct_run')
    root.mkdir(exist_ok=True)
    if mode == 'diagnostic':
        receipts = list(Path('/kaggle/input').rglob('run_receipt.json'))
        receipts = [p for p in receipts if (p.parent/'biohub_control/submission.csv').exists()]
        if len(receipts) != 1:
            raise RuntimeError('Attach the existing Biohub Lab Official Metric AB notebook output')
        previous = json.loads(receipts[0].read_text())
        manifest.update(datasets=previous['datasets'],
            scope='IN-SAMPLE: all four videos were in secondary detector training',
            prediction_sha256=previous['arms']['control']['sha256'],
            control_metrics=previous['arms']['control']['metrics'])
        data = root/'diagnostic_data'
        data.mkdir(exist_ok=True)
        for name in manifest['datasets']:
            for suffix in ('.zarr','.geff'):
                source = comp/'train'/f'{name}{suffix}'
                if not source.exists():
                    raise FileNotFoundError(source)
                destination = data/source.name
                if not destination.exists():
                    destination.symlink_to(source, target_is_directory=True)
        seed_csv = receipts[0].parent/'biohub_control/submission.csv'
        assert hashlib.sha256(seed_csv.read_bytes()).hexdigest() == manifest['prediction_sha256']
    else:
        data = comp/'test'
        seed_csv = Path('/kaggle/working/biohub_control/submission.csv')
    shapes = shapes_for(data)
    seeds = read_and_validate(seed_csv, shapes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model = load_hoct(assets/'general_v1.pt', device)
    config = HOCTConfig()
    results, statistics = {}, {}
    print('HOCT_CONFIG', json.dumps(asdict(config)), 'device',device,flush=True)
    for name in sorted(seeds):
        start = time.monotonic()
        nodes, _ = seeds[name]
        coords = np.array([[n[k] for k in ('t','z','y','x')] for _,n in sorted(nodes.items())], dtype=np.float32)
        images = zarr.open_group(str(data/f'{name}.zarr'), mode='r')['0']
        cache = root/f'{name}_morphology.npz'
        cache_key = hashlib.sha256(coords.tobytes() + json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()
        if cache.exists():
            with np.load(cache) as previous:
                if str(previous['cache_key']) != cache_key:
                    raise RuntimeError('Morphology cache belongs to different inputs/configuration')
                features, sizes = previous['features'], previous['sizes']
        else:
            features = np.empty((len(coords),19),dtype=np.float32)
            sizes = np.empty(len(coords),dtype=np.int32)
            for t in range(images.shape[0]):
                ids = np.flatnonzero(coords[:,0] == t)
                if len(ids):
                    features[ids],sizes[ids] = frame_features(images[t],coords[ids,1:],t,config)
                if t % 20 == 0:
                    print(name,'morphology frame',t,'seconds',round(time.monotonic()-start),flush=True)
            np.savez_compressed(cache,coords=coords,features=features,sizes=sizes,cache_key=cache_key)
        features_seconds = time.monotonic()-start
        edges = candidate_edges(coords,config)
        def progress(state):
            if state['window_start'] % 15 == 0:
                print(name,'HOCT',state,'seconds',round(time.monotonic()-start),flush=True)
        probabilities,orphans,inference_stats = score_edges(model,coords,features,edges,config,progress)
        selected = solve_lineage(coords,edges,probabilities,orphans,config)
        np.savez_compressed(root/f'{name}_scores.npz',coords=coords,candidates=edges,
            probabilities=probabilities,orphans=orphans,selected=selected)
        results[name] = (coords,selected)
        statistics[name] = {'seconds':time.monotonic()-start,'feature_seconds':features_seconds,
            'nodes':len(coords),'candidate_edges':len(edges),'selected_edges':len(selected),
            'division_sources':int((np.bincount(selected[:,0],minlength=len(coords)) == 2).sum()),
            'mask_median_voxels':float(np.median(sizes)), 'tiny_mask_fraction':float(np.mean(sizes<8)),
            **inference_stats}
        print('VIDEO_COMPLETE',name,json.dumps(statistics[name]),flush=True)
        write_csv(root/'predictions.csv',results)
        (root/'statistics.json').write_text(json.dumps(statistics,indent=2))
    output = root/'predictions.csv'
    read_and_validate(output,shapes)
    receipt = {'mode':mode,'implementation':'morphology_hoct_fixed_nodes_v1','config':asdict(config),
        'model_sha256':manifest['model_sha256'],'statistics':statistics,
        'scope':manifest['scope'] if mode=='diagnostic' else 'test inference; no measured score yet',
        'leaderboard_submitted':False}
    if mode == 'diagnostic':
        metrics = evaluate_csv(output,data)
        metrics['scope'] = manifest['scope']
        (root/'official_metrics.json').write_text(json.dumps(metrics,indent=2))
        receipt['metrics'] = metrics
        receipt['control'] = manifest['control_metrics']['summary']
        receipt['delta'] = {k:metrics['summary'][k]-receipt['control'][k]
                           for k in ('score','adj_edge_jaccard','division_jaccard')}
    else:
        import shutil
        shutil.copyfile(output,'/kaggle/working/submission.csv')
    receipt['prediction_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    (Path('/kaggle/working')/'hoct_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
