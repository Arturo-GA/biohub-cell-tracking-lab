"""Full frozen Harmonic image inference on calibration; defer metric to CPU."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from association_cpu_runner import read,save,sha
from notebook_runner import competition_dir
from biohub_lab.patch import patch_source

def main(package,root=Path('/kaggle/working/calibration_control')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E017-control',status='preparing',leaderboard_submitted=False,official_evaluation_pending=True)
    save(root/'result.json',result)
    try:
        names=read(package/'baseline/event_graph_split.json')['split']['calibration'];assert len(names)==16
        train=competition_dir()/'train';data=root/'images';data.mkdir(exist_ok=True)
        # Image-only directory: inference cannot discover sibling GT files here.
        for name in names:
            source=train/(name+'.zarr');assert source.exists()
            (data/source.name).symlink_to(source,target_is_directory=True)
        arm=root/'harmonic_control';arm.mkdir(exist_ok=True)
        source=patch_source((package/'baseline/harmonic_inference.py').read_text(),candidate=False)
        source=source.replace('/kaggle/working',str(arm));script=arm/'run.py';script.write_text(source)
        env=dict(os.environ,BIOHUB_LAB_DATA_DIR=str(data),BIOHUB_LAB_SRC=str(package/'src'),PYTHONPATH=str(package/'src'))
        result.update(status='image_inference',videos=names,source_sha256=sha(package/'baseline/harmonic_inference.py'))
        save(root/'result.json',result)
        subprocess.run([sys.executable,'-u',str(script)],cwd=arm,env=env,check=True)
        result.update(status='complete',seconds=time.monotonic()-start,csv_sha256=sha(arm/'submission.csv'),
            annotations_read=False,next_action='Validate export and evaluate frozen CSV in CPU notebook, paired with E016/E017')
        save(root/'result.json',result);print('CALIBRATION_CONTROL_READY',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
