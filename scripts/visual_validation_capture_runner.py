"""E023: reserved-video exact visual-evidence capture; volumetric inference needs GPU."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from association_cpu_runner import read,save,sha
from notebook_runner import competition_dir
from biohub_lab.patch import patch_source
from biohub_lab.visual_capture import inject_baseline

def main(package,root=Path('/kaggle/working/visual_validation_capture')):
    root.mkdir(exist_ok=True);start=time.monotonic()
    names=read(package/'baseline/e023_validation.json')['videos']
    result=dict(experiment='E023-capture',status='preparing',videos=names,annotations_read=False,leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        train=competition_dir()/'train';data=root/'images';data.mkdir(exist_ok=True)
        for name in names:(data/(name+'.zarr')).symlink_to(train/(name+'.zarr'),target_is_directory=True)
        arm=root/'harmonic_control';arm.mkdir(exist_ok=True)
        source=patch_source((package/'baseline/harmonic_inference.py').read_text(),candidate=False)
        # Apply capture after the immutable baseline finishes installing runtime patches.
        source=inject_baseline(source).replace('/kaggle/working',str(arm))
        script=arm/'run.py';script.write_text(source)
        env=dict(os.environ,BIOHUB_LAB_DATA_DIR=str(data),BIOHUB_LAB_SRC=str(package/'src'),PYTHONPATH=str(package/'src'))
        subprocess.run([sys.executable,'-u',str(script)],cwd=arm,env=env,check=True)
        cache=Path('/kaggle/working/visual_candidate_cache')
        assert {p.name for p in cache.iterdir() if p.is_dir()}==set(names)
        files={str(p.relative_to(cache)):sha(p) for p in sorted(cache.glob('*/*.npz'))}
        save(root/'capture_manifest.json',dict(files=files,annotations_read=False))
        result.update(status='complete',seconds=time.monotonic()-start,csv_sha256=sha(arm/'submission.csv'),pairs=len(files),official_evaluation_pending=True)
        save(root/'result.json',result);print('E023_CAPTURE_READY',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))

