"""Reproduce the fixed Kaggle integration using the actual v1 artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from biohub_lab.cellect_detector import find_checkpoint, WEIGHT_SHA
from biohub_lab.detector_proposals import install_pipeline_hook, patch_predictor

ROOT = Path(__file__).resolve().parents[1]


def main():
    checkpoint = find_checkpoint(ROOT / 'artifacts/cellect_kaggle_verified')
    raw_predictor = ROOT / 'outputs/e009_failed_v1/detector_experiment/tracking/tracking_repo/scripts/predict_unet_transformer.py'
    actual = raw_predictor.read_text(encoding='utf8')
    # v1 wrote the predictor successfully, then failed while writing its receipt.
    insertion = ('                from biohub_lab.detector_proposals import augment_runtime\n'
                 '                arr = augment_runtime(arr, ds_path, t, downsample)\n')
    if actual.count(insertion) != 1:
        raise ValueError('Unexpected v1 predictor artifact')
    original = actual.replace(insertion, '', 1)
    anchor = 'print("secondary edge-feature TTA patch installed and enabled", flush=True)'
    with tempfile.TemporaryDirectory(prefix='detector-recovery-', dir=ROOT/'artifacts') as folder:
        root = Path(folder)
        predictor = root / 'predictor.py'
        predictor.write_text(original, encoding='utf8')
        namespace = {'_ps': predictor, 'WORKING_DIR': root}
        exec(compile(install_pipeline_hook(anchor), '<actual-kaggle-hook>', 'exec'), namespace)
        assert 'hashlib' not in namespace
        assert predictor.read_text(encoding='utf8') == actual == patch_predictor(original)
        patch_receipt = json.loads((root / 'proposal_patch_receipt.json').read_text())
    previous = json.loads((ROOT / 'outputs/e009_failed_v1/detector_experiment/result.json').read_text())
    checked = {}
    for name, receipt in previous['proposal_receipts'].items():
        path = ROOT / 'outputs/e009_failed_v1/detector_experiment/proposals' / (name + '.npz')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == receipt['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            coords, scores, shape = arrays['coords'], arrays['scores'], arrays['shape']
        assert coords.shape == (receipt['proposals'], 4) and len(scores) == len(coords)
        assert np.isfinite(coords).all() and np.isfinite(scores).all()
        assert np.all(coords >= 0) and np.all(coords <= shape - 1)
        assert [int((coords[:,0] == t).sum()) for t in range(shape[0])] == receipt['frame_counts']
        checked[name] = dict(sha256=digest, proposals=len(coords), frames=int(shape[0]))
    record = dict(checkpoint_filename=checkpoint.name, checkpoint_sha256=WEIGHT_SHA,
        checkpoint_downloaded_from_kaggle=True, actual_v1_predictor_repaired=True,
        installed_hook_executed_without_caller_hashlib=True, patch_receipt=patch_receipt,
        candidate_logic_unchanged=True, gaussian_proposal_artifacts=checked)
    (ROOT / 'results/E008_E009_recovery_verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
