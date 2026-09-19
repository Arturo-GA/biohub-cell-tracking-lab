"""Run the production postprocessor on both real pilot outputs, CPU only."""
import json,time
from pathlib import Path
import torch
from association_cpu_runner import locate,read,sha,save
from visual_submission_postprocess import postprocess

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/visual_submission_smoke')):
    root.mkdir(exist_ok=True);start=time.monotonic()
    if torch.cuda.is_available():raise RuntimeError('CPU only')
    control=locate(input_root,'visual_capture/result.json').parent
    previous=locate(input_root,'visual_exact/frozen_predictions.json').parent
    names=read(package/'baseline/e022_capture.json')['videos']
    train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
    images=root/'images';images.mkdir(exist_ok=True)
    for name in names:(images/(name+'.zarr')).symlink_to(train/(name+'.zarr'),target_is_directory=True)
    result=postprocess(control/'harmonic_control',control.parent/'visual_candidate_cache',images,root/'submission.csv',root/'postprocess.json')
    expected=read(previous/'frozen_predictions.json')['csv_sha256']['visual']
    assert result['csv_sha256']==expected
    save(root/'result.json',dict(status='complete',seconds=time.monotonic()-start,production_csv_exactly_matches_pilot=True,
        csv_sha256=expected,annotations_read=False,accelerator='none',leaderboard_submitted=False))
    print('PRODUCTION_POSTPROCESS_VERIFIED',expected,flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
