"""Portable E013 commands. Kaggle stages are CPU-only; CUDA is local only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from biohub_lab.event_data import save_json, sparse_labels
from biohub_lab.event_portable import load_portable, prepare_mmap, sha, signature
from biohub_lab.event_train import load_arrays


def read(path):
    return json.loads(Path(path).read_text())


def device_policy(device):
    if os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION'):
        raise RuntimeError('This workflow is configured for the laptop and Kaggle CPU')
    if str(device).startswith('cuda') and (Path('/kaggle').is_dir() or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')):
        raise RuntimeError('Kaggle GPU disabled by the project compute policy')


def verify_frozen(root, split):
    frozen = read(root/'frozen_inputs.json')
    if frozen.get('annotations_read') is not False or set(frozen['records']) != set(sum((split[k] for k in ('fit','calibration','evaluation')), [])):
        raise ValueError('Incomplete image-only input freeze')
    for name, record in frozen['records'].items():
        if record.get('annotations_read') is not False or record.get('harmonic_associations_used', False):
            raise ValueError('Invalid image-only generation provenance')
        for f, digest in record['files'].items():
            if sha(root/'videos'/name/f) != digest:
                raise ValueError('Frozen input changed: '+name+'/'+f)
    return frozen


def freeze(root, split):
    records = {}; pins = read(ROOT/'baseline/event_graph_inputs.json')
    for name in sum((split[k] for k in ('fit','calibration','evaluation')), []):
        folder = root/'videos'/name; receipt = read(folder/'generation.json')
        if receipt['annotations_read'] or receipt.get('harmonic_associations_used', False):
            raise ValueError('Invalid generation provenance')
        if set(receipt['files']) != {'graph.npz', 'features.npz'}:
            raise ValueError('Incomplete generation receipt')
        if any(sha(folder/f) != d for f,d in receipt['files'].items()):
            raise ValueError('Generation checksum mismatch')
        if name in split['evaluation'] and sha(folder/'graph.npz') != pins['files'][name]['combined/graph.npz']:
            raise ValueError('E012 evaluation graph changed')
        records[name] = receipt
    value = dict(annotations_read=False, records=records, split_sha256=signature(split))
    destination = root/'frozen_inputs.json'
    if destination.exists() and read(destination) != value:
        raise ValueError('Existing freeze differs; preserve it and version the experiment')
    save_json(destination, value)
    return dict(status='frozen', videos=len(records))


def run(args):
    device_policy(args.device)
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    split = read(args.split)['split']; names = split[args.role] if args.role != 'all' else sum((split[k] for k in ('fit','calibration','evaluation')), [])
    if args.video:
        if args.video not in names:
            raise ValueError('Video is outside the selected partition')
        names = [args.video]
    if args.stage == 'inventory':
        report = {k: {n: [f for f in ('graph.npz','features.npz','generation.json') if not (root/'videos'/n/f).is_file()]
                      for n in split[k]} for k in ('fit','calibration','evaluation')}
        save_json(root/'readiness.json', report)
        return {k: dict(total=len(v), complete=sum(not missing for missing in v.values())) for k,v in report.items()}
    if args.stage == 'freeze':
        return freeze(root, split)
    if args.stage in ('labels','pack','train','calibrate','score'):
        frozen = verify_frozen(root, split)
    if args.stage == 'labels':
        if args.role not in ('fit','calibration') or not args.data_dir:
            raise ValueError('Labels require a fit/calibration role and a data directory')
        import numpy as np
        from biohub_lab.temporal_data import load_gt
        counts = {}
        for name in names:
            folder = root/'videos'/name; graph = load_arrays(folder/'graph.npz')
            truth, edges = load_gt(Path(args.data_dir)/(name+'.geff'))
            labels, counts[name] = sparse_labels(graph, truth, edges)
            np.savez_compressed(folder/'labels.npz', **labels)
            save_json(folder/'labels.json', dict(counts=counts[name], graph_sha256=sha(folder/'graph.npz'),
                frozen_inputs_sha256=sha(root/'frozen_inputs.json'), labels_sha256=sha(folder/'labels.npz')))
        return dict(status='labels_ready', videos=len(counts))
    if args.stage == 'pack':
        for name in names:
            prepare_mmap(root/'videos'/name, labels=name not in split['evaluation'])
        return dict(status='portable_arrays_ready', videos=len(names))
    if args.stage == 'train':
        for name in split['fit']+split['calibration']:
            folder=root/'videos'/name; labels=read(folder/'labels.json')
            if labels['graph_sha256'] != sha(folder/'graph.npz') or labels.get('labels_sha256') != sha(folder/'labels.npz'):
                raise ValueError('Supervision is not bound to the frozen graph and label arrays')
        from biohub_lab.event_local_train import train
        return train(root, split, args.device, resume=args.resume, stop_after=args.stop_after, time_budget=args.time_budget)
    if args.stage == 'calibrate':
        from biohub_lab.event_local_train import calibrate
        return calibrate(root, split, args.device)
    if args.stage == 'score':
        import numpy as np
        from biohub_lab.event_inference import load_model, score_graph
        model = load_model(root/'best.pt', args.device); calibration = read(root/'calibration.json')
        if sha(root/'best.pt') != calibration['checkpoint_sha256']:
            raise ValueError('Calibration checkpoint changed')
        for name in names:
            folder = root/'videos'/name; receipt_path = folder/'event_scores.json'
            identity = dict(checkpoint_sha256=sha(root/'best.pt'), calibration_sha256=sha(root/'calibration.json'),
                            inputs=frozen['records'][name]['files'])
            if receipt_path.exists():
                old = read(receipt_path)
                if old['identity'] == identity and sha(folder/'event_scores.npz') == old['scores_sha256']:
                    continue
                raise ValueError('Existing scoring output differs')
            video = load_portable(folder, labels=False)
            scored, counts = score_graph(model, video, calibration, args.device, stream_chunk=256)
            np.savez_compressed(folder/'event_scores.npz', **scored)
            save_json(receipt_path, dict(identity=identity, counts=counts, scores_sha256=sha(folder/'event_scores.npz')))
            del video, scored
        return dict(status='scored', videos=len(names))
    if args.stage == 'solve':
        import numpy as np
        from biohub_lab.event_solver import select_graph
        for name in names:
            folder = root/'videos'/name; receipt = read(folder/'event_scores.json')
            if sha(folder/'event_scores.npz') != receipt['scores_sha256'] or sha(folder/'graph.npz') != receipt['identity']['inputs']['graph.npz']:
                raise ValueError('Scoring inputs changed before CPU selection')
            graph = load_arrays(folder/'graph.npz'); scored = load_arrays(folder/'event_scores.npz')
            selected, solver = select_graph(graph, **scored)
            np.savez_compressed(folder/'prediction.npz', coords=graph['coords'], edges=selected, shape=graph['shape'])
            save_json(folder/'prediction.json', dict(receipt['counts'], solver=solver, scores_sha256=receipt['scores_sha256'], evaluation_annotations_read=False))
        return dict(status='selected', videos=len(names))
    if args.stage == 'csv':
        from biohub_lab.event_inference import write_csv
        from biohub_lab.submission import read_and_validate
        if args.role != 'evaluation' or args.video:
            raise ValueError('Final CSV must contain the complete evaluation partition')
        destination = root/'candidate.csv'; write_csv(root, names, destination)
        shapes = {n: load_arrays(root/'videos'/n/'graph.npz')['shape'] for n in names}
        read_and_validate(destination, shapes)
        save_json(root/'frozen_predictions.json', dict(csv_sha256=sha(destination), checkpoint_sha256=sha(root/'best.pt'),
            calibration_sha256=sha(root/'calibration.json'), evaluation_annotations_read=False,
            predictions={n:sha(root/'videos'/n/'prediction.npz') for n in names}))
        return dict(status='csv_frozen', videos=len(names), csv_sha256=sha(destination))
    if args.stage == 'evaluate':
        from biohub_lab.evaluate import evaluate_csv
        pins = read(root/'frozen_predictions.json')
        if sha(root/'candidate.csv') != pins['csv_sha256'] or not args.data_dir:
            raise ValueError('Evaluation needs frozen CSV and exact cohort data directory')
        metrics = evaluate_csv(root/'candidate.csv', args.data_dir)
        metrics['scope'] = 'Conditional selector development on E012 cohort; not independent end-to-end validation'
        save_json(root/'candidate_metrics.json', metrics)
        return metrics['summary']
    raise ValueError(args.stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory','freeze','labels','pack','train','calibrate','score','solve','csv','evaluate'])
    parser.add_argument('--root', required=True); parser.add_argument('--split', default=str(ROOT/'baseline/event_graph_split.json'))
    parser.add_argument('--role', choices=['all','fit','calibration','evaluation'], default='all')
    parser.add_argument('--video'); parser.add_argument('--device', default='cpu'); parser.add_argument('--data-dir')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--stop-after', type=int); parser.add_argument('--time-budget', type=float)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
