"""Offline Kaggle submission: frozen Harmonic image inference plus global links."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from biohub_lab.patch import patch_source
from biohub_lab.visual_capture import inject_baseline

def main(package):
    root=Path('/kaggle/working/visual_submission');root.mkdir(exist_ok=True);start=time.monotonic()
    record=dict(status='image_inference',leaderboard_submitted=False,annotations_read=False)
    def save(): (root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        test=next(p/'test' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'test').is_dir())
        control=root/'harmonic_control';control.mkdir(exist_ok=True)
        source=inject_baseline(patch_source((package/'baseline/harmonic_inference.py').read_text(),candidate=False))
        source=source.replace('/kaggle/working',str(control));script=control/'run.py';script.write_text(source)
        env=dict(os.environ,BIOHUB_LAB_DATA_DIR=str(test),BIOHUB_LAB_SRC=str(package/'src'),PYTHONPATH=str(package/'src'))
        subprocess.run([sys.executable,'-u',str(script)],cwd=control,env=env,check=True)
        record['status']='cpu_association';save()
        # Fresh interpreter after baseline installs its pinned offline dependencies.
        subprocess.run([sys.executable,'-u',str(package/'scripts/visual_submission_postprocess.py'),
            '--control',str(control),'--cache','/kaggle/working/visual_candidate_cache','--test',str(test),
            '--output','/kaggle/working/submission.csv','--receipt',str(root/'postprocess.json')],env=env,check=True)
        target=Path('/kaggle/working/submission.csv')
        record.update(status='complete',seconds=time.monotonic()-start,csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),submission_ready=True)
        save();print('VISUAL_SUBMISSION_READY',record,flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error));save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
