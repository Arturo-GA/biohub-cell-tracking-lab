"""Recompute hidden-test images with the selected jointly evaluated pipeline."""
import json,os,subprocess,sys,time
from pathlib import Path
from biohub_lab.patch import patch_source
from biohub_lab.public_postprocess import fast_checkpoint_paths
from biohub_lab.visual_capture import inject_baseline

def main(package,config_name='e054_submission.json'):
    start=time.monotonic();out=Path('/kaggle/working/joint_submission');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline'/config_name).read_text())
    test=next(p/'test' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'test').exists())
    control=out/'harmonic';control.mkdir(exist_ok=True)
    source=fast_checkpoint_paths(patch_source((package/'baseline/harmonic_inference.py').read_text(),candidate=False))
    if cfg['visual']:source=inject_baseline(source)
    anchor='write_test_submission("base")';assert source.count(anchor)==1
    settings=f"MOTION_RELINK_TIGHT_UM = {cfg['radius']!r}\nDEEPCENTER_SAFE_DIV_THRESHOLD = {cfg['division_threshold']!r}\n"
    source=source.replace(anchor,settings+anchor).replace('/kaggle/working',str(control))
    script=control/'run.py';script.write_text(source)
    env=dict(os.environ,BIOHUB_LAB_DATA_DIR=str(test),BIOHUB_LAB_SRC=str(package/'src'),PYTHONPATH=str(package/'src'))
    for p in (Path('/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt'),Path('/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')):
        if p.is_file():
            env['BIOHUB_DEEPCENTER_CHECKPOINT']=str(p);break
    subprocess.run([sys.executable,'-u',str(script)],cwd=control,env=env,check=True)
    subprocess.run([sys.executable,'-u',str(package/'scripts/joint_submission_finish.py'),str(package),str(test),str(control),config_name],env=env,check=True)
    print('ENSEMBLE_JOINT_SUBMISSION_COMPLETE',cfg['selected'],time.monotonic()-start,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
