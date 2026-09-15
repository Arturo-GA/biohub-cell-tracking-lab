"""Freeze filename-only E013 split and image-derived E012 graph digests."""
import hashlib
import json
from pathlib import Path
from biohub_lab.event_data import select_split, save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    dev = json.loads((ROOT/'baseline/detection_dag_dev.json').read_text())
    inventory = json.loads((ROOT/'outputs/temporal_prepare/temporal_cache/cache_manifest.json').read_text())
    names = sorted(v['name'] for v in inventory['videos'])
    digest = hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()
    assert digest == dev['training_names_sha256']
    split = select_split(names, dev['development'], dev['excluded'])
    save_json(ROOT/'baseline/event_graph_split.json', dict(experiment='E013', split=split,
        selection='SHA256 of seed and filename within acquisition group; labels and image content not used',
        training_names_sha256=digest, config_sha256='e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a'))
    cached = ROOT/'outputs/e012_v1/harmonic_dag_experiment'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pins = dict(experiment='E013', slug='biohub-lab-harmonic-detection-dag', version=1,
        frozen_sha256=sha(cached/'frozen_inputs.json'), files={
            n: {'combined/graph.npz': sha(cached/'videos'/n/'combined/graph.npz')} for n in split['evaluation']})
    save_json(ROOT/'baseline/event_graph_inputs.json', pins)
    print({key: len(value) for key, value in split.items() if isinstance(value, list)})


if __name__ == '__main__':
    main()
