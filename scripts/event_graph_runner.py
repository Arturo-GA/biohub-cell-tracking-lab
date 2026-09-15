"""E013: detector-space graph learning, calibrated selection and paired evaluation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from notebook_runner import competition_dir
from biohub_lab.event_data import extract_visual, neighborhoods, save_json, select_split, sparse_labels
from biohub_lab.event_train import load_arrays
from biohub_lab.harmonic_centers import CHECKPOINTS, DETECTOR_CONFIG, detector_source, merge_primary, save_centers
from biohub_lab.detection_dag import build_dag, save_dag


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_cached(package, input_root):
    pins = read(Path(package)/'baseline/event_graph_inputs.json')
    roots = [Path(input_root)/pins['slug'], Path(input_root)/'notebooks/jarturo'/pins['slug']]
    matches = [p/'harmonic_dag_experiment' for p in roots if (p/'harmonic_dag_experiment/frozen_inputs.json').is_file()]
    if len(matches) != 1:
        raise ValueError('Attach exactly one pinned E012 output')
    cached = matches[0]
    if sha(cached/'frozen_inputs.json') != pins['frozen_sha256']:
        raise ValueError('E012 frozen manifest changed')
    frozen = read(cached/'frozen_inputs.json')
    if frozen['annotations_read'] or not frozen['all_detection_graphs_complete']:
        raise ValueError('E012 generation provenance is invalid')
    for name, files in pins['files'].items():
        for suffix, digest in files.items():
            if sha(cached/'videos'/name/suffix) != digest:
                raise ValueError('E012 cached file changed: '+name+'/'+suffix)
    return cached, pins


def generate_worker(data, root, names, cached, runtime_path, index):
    import torch
    import zarr
    from biohub_lab.cellect_detector import run_video as cellect
    from biohub_lab.gaussian_detector import run_video as gaussian
    torch.set_num_threads(2); data = Path(data); root = Path(root); cached = Path(cached)
    runtime = read(runtime_path); device = torch.device('cuda')
    spec = importlib.util.spec_from_file_location('event_image_detector', runtime['module'])
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    primary, window, ds = module.load_model(Path(runtime['primary']), device)
    secondary, window2, ds2 = module.load_model(Path(runtime['secondary']), device)
    if window != 2 or window != window2 or tuple(ds) != (1, 4, 4) or tuple(ds) != tuple(ds2):
        raise ValueError('Frozen model shape changed')
    cfg = module.PredictConfig(det_threshold=.965, det_tta=True, pool_kernel_um=3.)
    progress = dict(completed=[], worker=index, status='generating', annotations_read=False)
    progress_path = root/f'worker_{index}.json'
    try:
        for name in names:
            start = time.monotonic(); folder = root/'videos'/name; folder.mkdir(parents=True, exist_ok=True)
            progress['current'] = name; save_json(progress_path, progress)
            receipt_path = folder/'generation.json'
            if receipt_path.is_file():
                receipt = read(receipt_path)
                if (receipt['runtime_sha256'] == sha(runtime_path) and not receipt['annotations_read'] and
                    all(sha(folder/f) == digest for f, digest in receipt['files'].items())):
                    progress['completed'].append(name); continue
                raise ValueError('Existing generation stage failed integrity checks')
            image = data/(name+'.zarr'); shape = zarr.open_group(str(image), mode='r')['0'].shape
            reuse = (cached/'videos'/name/'combined/graph.npz').is_file()
            if reuse:
                shutil.copyfile(cached/'videos'/name/'combined/graph.npz', folder/'graph.npz')
                graph = load_arrays(folder/'graph.npz')
                if not np.array_equal(graph['shape'], shape):
                    raise ValueError('Cached image shape changed')
            else:
                aux = []
                for method, detector in [('cellect', cellect), ('gaussian', gaussian)]:
                    path = folder/(method+'.npz')
                    result = detector(image, path, weights=runtime['cellect']) if method == 'cellect' else detector(image, path)
                    if sha(path) != result['sha256']:
                        raise ValueError('Auxiliary detector checksum mismatch')
                    values = load_arrays(path)
                    if not np.array_equal(values['shape'], shape):
                        raise ValueError('Auxiliary image shape changed')
                    aux.append((values['coords'], values['scores']))
                coords, associations = module.predict_video(primary, image, device, cfg, window_size=window,
                    downsample=ds, secondary_model=secondary, secondary_detection_weight=.80)
                if associations:
                    raise ValueError('Detector returned an association')
                save_centers(folder/'harmonic.npz', coords, shape, dict(checkpoint_sha256=CHECKPOINTS))
                points, scores, origin = merge_primary(coords, aux, shape)
                graph = build_dag(points, scores, origin, shape)
                save_dag(graph, folder, name)
            visual = extract_visual(module, [primary, secondary], image, graph['coords'], device)
            neighbors = neighborhoods(graph['coords'])
            np.savez_compressed(folder/'features.npz', visual=visual, neighbors=neighbors)
            receipt = dict(dataset=name, annotations_read=False, harmonic_associations_used=False,
                cached_e012_graph=reuse, runtime_sha256=sha(runtime_path), nodes=len(graph['coords']),
                shape=list(map(int, shape)), visual_shape=list(visual.shape), seconds=time.monotonic()-start,
                files={f: sha(folder/f) for f in ('graph.npz', 'features.npz')})
            save_json(receipt_path, receipt); progress['completed'].append(name); save_json(progress_path, progress)
            print('EVENT_FEATURES_READY', name, json.dumps(receipt), flush=True)
        progress.update(status='complete'); progress.pop('current', None); save_json(progress_path, progress)
    except Exception as error:
        progress.update(status='failed', error=str(error)); save_json(progress_path, progress); raise


def workers(data, root, names, cached, runtime, repo):
    import torch
    count = min(2, torch.cuda.device_count())
    if not count:
        raise RuntimeError('E013 requires CUDA')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    tokens = visible.split(',') if visible and visible != '-1' else list(map(str, range(count)))
    processes = []
    try:
        for index in range(count):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=tokens[index], OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='1', BIOHUB_EDGE_FEATURE_TTA='0', BIOHUB_SECONDARY_EDGE_FEATURE_TTA='0',
                BIOHUB_DIAGNOSTIC_ARM='', BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION='.90')
            env['PYTHONPATH'] = str(Path(repo)/'src')+os.pathsep+env['PYTHONPATH']
            command = [sys.executable, '-u', __file__, 'worker', str(data), str(root), json.dumps(names[index::count]),
                       str(cached), str(runtime), str(index)]
            processes.append(subprocess.Popen(command, env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('E013 feature worker failed; inspect worker receipt')
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('E013 feature worker failed')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate(); process.wait()


def evaluate_dir(data, names, dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        for suffix in ('.zarr', '.geff'):
            target = Path(data)/(name+suffix); link = dest/target.name
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
    return dest


def run_control(package, data, root):
    from biohub_lab.patch import patch_source
    arm = Path(root)/'harmonic_control'; arm.mkdir(exist_ok=True)
    source = patch_source((Path(package)/'baseline/harmonic_inference.py').read_text(), candidate=False)
    source = source.replace('/kaggle/working', str(arm))
    source += '\nfrom biohub_lab.evaluate import evaluate_csv\n'
    source += 'lab_metrics=evaluate_csv(SUBMISSION_PATH, TEST_DIR)\n'
    source += 'Path(WORKING_DIR / "official_metrics.json").write_text(json.dumps(lab_metrics, indent=2))\n'
    script = arm/'run.py'; script.write_text(source)
    env = dict(os.environ, BIOHUB_LAB_DATA_DIR=str(data), BIOHUB_LAB_SRC=str(Path(package)/'src'),
               PYTHONPATH=str(Path(package)/'src'))
    # Generation workers changed only their own environments. Full baseline TTA
    # is configured by the frozen baseline source in this separate interpreter.
    subprocess.run([sys.executable, '-u', str(script)], cwd=arm, env=env, check=True)
    return read(arm/'official_metrics.json')


def main(package, repo, primary, secondary, root=Path('/kaggle/working/event_graph_experiment'), input_root=Path('/kaggle/input')):
    from biohub_lab.cellect_detector import find_checkpoint
    from biohub_lab.temporal_data import load_gt
    from biohub_lab.event_train import train_and_calibrate
    from biohub_lab.event_inference import load_model, infer_video, write_csv
    from biohub_lab.evaluate import evaluate_csv
    package = Path(package); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); receipt_path = root/'result.json'
    result = dict(experiment='E013', status='verifying_inputs', neural_training=True, leaderboard_submitted=False,
        scope='Selector fitting and calibration use separate videos. Evaluation reuses the 48 E012 design-development videos. '
              'Public secondary detector trained on all 199 train videos: conditional development evidence, not independent end-to-end validation.')
    save_json(receipt_path, result)
    try:
        cached, pins = verify_cached(package, input_root)
        manifest = read(package/'baseline/event_graph_split.json'); split = manifest['split']
        comp = competition_dir(); data = comp/'train'; names = sorted(p.stem for p in data.glob('*.zarr'))
        if hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest() != manifest['training_names_sha256']:
            raise ValueError('Training inventory changed')
        if select_split(names, split['evaluation'], split['excluded'], split['seed']) != split:
            raise ValueError('Filename-only split changed')
        if split['evaluation'] != sorted(pins['files']):
            raise ValueError('E012 evaluation cohort changed')
        all_names = split['fit']+split['calibration']+split['evaluation']
        if set(all_names) & {p.stem for p in (comp/'test').glob('*.zarr')}:
            raise ValueError('Visible test video included')
        weights = dict(primary=str(primary), secondary=str(secondary))
        for key, path in weights.items():
            if sha(path) != CHECKPOINTS[key]:
                raise ValueError('Frozen public weight changed')
            if sha(Path(path).with_name('config.json')) != manifest['config_sha256']:
                raise ValueError('Frozen public model configuration changed')
        module = root/'image_detector.py'
        module.write_text(detector_source((Path(repo)/'scripts/predict_unet_transformer.py').read_text(),
                                          (Path(repo)/'scripts/train_unet_transformer.py').read_text()))
        runtime = dict(weights, cellect=str(find_checkpoint(input_root)), module=str(module),
                       detector_source_sha256=sha(module), checkpoint_sha256=CHECKPOINTS,
                       detector_config=DETECTOR_CONFIG, feature_mode='single_view_first_seen_dual_UNet_32_plus_32_trilinear')
        runtime_path = root/'runtime.json'; save_json(runtime_path, runtime)
        result.update(status='generating_detector_graphs_and_visual_features', split=split,
                      split_manifest_sha256=sha(package/'baseline/event_graph_split.json'),
                      e012_input_pins_sha256=sha(package/'baseline/event_graph_inputs.json'), runtime=runtime)
        save_json(receipt_path, result); workers(data, root, all_names, cached, runtime_path, repo)
        frozen = {}
        for name in all_names:
            folder = root/'videos'/name; record = read(folder/'generation.json')
            if record['annotations_read'] or any(sha(folder/f) != s for f, s in record['files'].items()):
                raise ValueError('Generation provenance/checksum changed')
            frozen[name] = record
        save_json(root/'frozen_inputs.json', dict(annotations_read=False, records=frozen))
        result.update(status='building_sparse_training_labels', frozen_inputs_sha256=sha(root/'frozen_inputs.json'),
                      generation_seconds=time.monotonic()-start); save_json(receipt_path, result)
        counts = {}
        # Evaluation GEFF remains unopened until all final candidate predictions freeze.
        for name in split['fit']+split['calibration']:
            folder = root/'videos'/name; graph = load_arrays(folder/'graph.npz')
            truth, edges = load_gt(data/(name+'.geff')); labels, counts[name] = sparse_labels(graph, truth, edges)
            np.savez_compressed(folder/'labels.npz', **labels)
            save_json(folder/'labels.json', dict(counts=counts[name], graph_sha256=sha(folder/'graph.npz')))
        save_json(root/'supervision_counts.json', counts)
        result.update(status='training_graph_selector', supervision_counts=counts); save_json(receipt_path, result)
        calibration = train_and_calibrate(root, split)
        result.update(status='predicting_evaluation_graphs', calibration=calibration); save_json(receipt_path, result)
        model = load_model(root/'best.pt', 'cuda')
        for name in split['evaluation']:
            folder = root/'videos'/name
            if any(sha(folder/f) != s for f, s in frozen[name]['files'].items()):
                raise ValueError('Evaluation inputs changed')
            stats = infer_video(root, name, model, calibration)
            print('EVENT_PREDICTION_READY', name, json.dumps(stats), flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
        csv_path = root/'candidate.csv'; write_csv(root, split['evaluation'], csv_path)
        prediction_pins = dict(csv_sha256=sha(csv_path), checkpoint_sha256=sha(root/'best.pt'),
            calibration_sha256=sha(root/'calibration.json'), evaluation_annotations_read=False,
            predictions={n: sha(root/'videos'/n/'prediction.npz') for n in split['evaluation']})
        save_json(root/'frozen_predictions.json', prediction_pins)
        eval_data = evaluate_dir(data, split['evaluation'], root/'evaluation_data')
        candidate = evaluate_csv(csv_path, eval_data); candidate['scope'] = result['scope']
        save_json(root/'candidate_metrics.json', candidate)
        if sha(csv_path) != prediction_pins['csv_sha256']:
            raise ValueError('Final CSV changed during evaluation')
        result.update(status='complete', candidate=candidate, seconds=time.monotonic()-start,
            paired_comparison_pending=True, paired_control_kernel='jarturo/biohub-lab-event-graph-control',
            next_action='Compare with the separate E013 full-Harmonic control on these same 48 videos; review false divisions and solver fallbacks before test submission.')
        save_json(receipt_path, result); print('EVENT_GRAPH_RESULT', json.dumps(result, indent=2), flush=True)
    except Exception as error:
        result.update(failed_stage=result['status'], status='failed', error=str(error), seconds=time.monotonic()-start)
        save_json(receipt_path, result); (root/'error.txt').write_text(traceback.format_exc()); raise


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        generate_worker(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7]))
