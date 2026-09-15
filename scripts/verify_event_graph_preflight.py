"""Real checkpoint feature extraction and end-to-end graph mechanics on synthetic data."""
import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

import numpy as np
import torch
import zarr
from biohub_lab.event_data import extract_visual, neighborhoods, node_inputs, save_json
from biohub_lab.event_model import EventGraphNet
from biohub_lab.event_train import load_arrays
from biohub_lab.event_inference import score_graph, write_csv
from biohub_lab.event_solver import select_graph
from biohub_lab.harmonic_centers import CHECKPOINTS, detector_source
from biohub_lab.submission import read_and_validate

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'artifacts/e012_detector_check'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2); torch.manual_seed(13013)
    work = ROOT/'artifacts/e013_preflight'; work.mkdir(exist_ok=True)
    for name, digest in CHECKPOINTS.items():
        assert sha(ASSETS/name/'edge_predictor_best.pth') == digest
    source = detector_source((ASSETS/'smoke/repo/scripts/predict_unet_transformer.py').read_text(),
                             (ASSETS/'repo/scripts/train_unet_transformer.py').read_text())
    module_path = work/'image_detector.py'; module_path.write_text(source.replace('/kaggle/working', work.as_posix()))
    sys.path.insert(0, str(ASSETS/'repo/src'))
    spec = importlib.util.spec_from_file_location('e013_real_detector', module_path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    models = [module.load_model(ASSETS/name/'edge_predictor_best.pth', torch.device('cpu'))[0] for name in ('primary', 'secondary')]
    graph = load_arrays(ASSETS/'smoke/result/videos/synthetic/combined/graph.npz')
    image = ASSETS/'smoke/synthetic.zarr'
    opened = []; original = zarr.open_group
    def image_only(path, *args, **kwargs):
        assert str(path).endswith('.zarr'), 'Non-image read during feature extraction'
        opened.append(str(path)); return original(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pretrained association executed')
    for model in models:
        model.predict_edges = forbidden
    with patch.object(zarr, 'open_group', side_effect=image_only):
        visual = extract_visual(module, models, image, graph['coords'], 'cpu')
    assert visual.shape == (len(graph['coords']), 64) and np.isfinite(visual).all()
    assert visual.std() > 0 and not np.array_equal(visual[:, :32], visual[:, 32:])
    # Feature coordinates displaced by one native XY pixel must not collapse to
    # integer division by four on the downsampled feature grid.
    shifted = graph['coords'].copy(); shifted[:, 3] = np.minimum(shifted[:, 3]+1, graph['shape'][3]-1)
    with patch.object(zarr, 'open_group', side_effect=image_only):
        other = extract_visual(module, models, image, shifted, 'cpu')
    assert not np.array_equal(visual, other)
    video = dict(graph=graph, inputs=node_inputs(graph, visual), neighbors=neighborhoods(graph['coords']))
    model = EventGraphNet().eval()
    # Synthetic mechanics, not a trained checkpoint or an accuracy estimate.
    scores, counts = score_graph(model, video, {'edge': {'logit': -3.}, 'division': {'logit': 10.}}, device='cpu')
    selected, solver = select_graph(graph, **scores)
    assert len(selected) and not solver['fallback_windows']
    folder = work/'videos/synthetic'; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'prediction.npz', coords=graph['coords'], edges=selected, shape=graph['shape'])
    write_csv(work, ['synthetic'], work/'candidate.csv')
    read_and_validate(work/'candidate.csv', {'synthetic': graph['shape']})
    pins = json.loads((ROOT/'baseline/event_graph_inputs.json').read_text())
    cached = ROOT/'outputs/e012_v1/harmonic_dag_experiment'
    assert sha(cached/'frozen_inputs.json') == pins['frozen_sha256']
    for name, files in pins['files'].items():
        for suffix, digest in files.items():
            assert sha(cached/'videos'/name/suffix) == digest
    digests = {}
    for arm in ('event_graph', 'event_graph_control'):
        nb = json.loads((ROOT/'kaggle'/arm/'notebook.ipynb').read_text())
        code = ''.join(nb['cells'][1]['source']); parsed = ast.parse(code)
        assignment = next(n for n in parsed.body if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        payload = base64.b64decode(ast.literal_eval(assignment.value.args[0])); digest = hashlib.sha256(payload).hexdigest()
        assert digest == json.loads((ROOT/'kaggle'/arm/'payload.json').read_text())['sha256']
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for path in archive.namelist():
                assert archive.read(path) == (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
                assert not path.startswith(('outputs/', 'artifacts/'))
            if arm == 'event_graph':
                extracted = work/'package'; extracted.mkdir(exist_ok=True)
                assert all((extracted/n).resolve().is_relative_to(extracted.resolve()) for n in archive.namelist())
                archive.extractall(extracted)
                env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(extracted/'src'), str(extracted/'scripts'),
                                                                  os.environ.get('PYTHONPATH', '')]))
                code = 'import event_graph_runner,event_control_runner; from biohub_lab.event_inference import score_graph; '
                # Official evaluation imports polars/tracksdata only after the
                # frozen Kaggle bootstrap installs its pinned support wheels.
                code += 'from pathlib import Path; '
                code += f'assert Path(event_graph_runner.__file__).is_relative_to(Path({str(extracted)!r})); '
                code += f'assert Path(event_control_runner.__file__).is_relative_to(Path({str(extracted)!r}))'
                subprocess.run([sys.executable, '-X', 'utf8', '-c', code], env=env, cwd=extracted, check=True)
        digests[arm] = digest
    receipt = dict(experiment='E013', payload_sha256=digests, actual_public_checkpoints_verified=CHECKPOINTS,
        actual_dual_UNet_cpu_features=True, association_calls_blocked=True, annotation_access_blocked=True,
        subvoxel_sampling_changes_features=True, feature_shape=list(visual.shape),
        all_eligible_pairs_scored=counts['scored_pairs'], selected_edges=len(selected),
        feasible_temporal_milp=True, final_integer_csv_valid=True, cached_e012_graphs_verified=48,
        all_packaged_files_verified=True, extracted_minimal_package_imports_verified=True,
        official_metric_executed_locally=False, metric_dependencies_provided_by_kaggle_support=True,
        scope='Technical test on synthetic images; random graph model. No accuracy evidence.')
    save_json(ROOT/'results/E013_preflight.json', receipt); print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
