"""Self-contained hidden-test inference; no cached test predictions."""
import json,os,subprocess,sys,time
from pathlib import Path
from biohub_lab.patch import patch_source

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/weighted_submission');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e050_submission.json').read_text())
    test=next(p/'test' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'test').exists())
    control=out/'harmonic';control.mkdir(exist_ok=True)
    source=patch_source((package/'baseline/harmonic_inference.py').read_text(),candidate=False)
    anchor='write_test_submission("base")'
    assert source.count(anchor)==1
    # Actual public V3/proxy output selects tight55. Fix it at inference time;
    # do not rerun their training-data sweep inside a competition submission.
    source=source.replace(anchor,'MOTION_RELINK_TIGHT_UM = 5.5\n'+anchor)
    source=source.replace('/kaggle/working',str(control))
    script=control/'run.py';script.write_text(source)
    env=dict(os.environ,BIOHUB_LAB_DATA_DIR=str(test),BIOHUB_LAB_SRC=str(package/'src'),PYTHONPATH=str(package/'src'))
    subprocess.run([sys.executable,'-u',str(script)],cwd=control,env=env,check=True)
    subprocess.run([sys.executable,'-u',str(package/'scripts/weighted_submission_finish.py'),str(package),str(test),str(control)],env=env,check=True)
    print('ENSEMBLE_SUBMISSION_COMPLETE',cfg['mode'],time.monotonic()-start,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
