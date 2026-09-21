"""Use exactly the visual assignment and bridge algorithms compared in E053."""
import json,shutil,sys
from pathlib import Path
import zarr
from ensemble_evaluate_runner import sha
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import read_and_validate

def main(package,test,control,config_name='e054_submission.json'):
    out=Path('/kaggle/working/joint_submission');cfg=json.loads((package/'baseline'/config_name).read_text())
    if cfg['visual']:
        from visual_submission_postprocess import postprocess
        graph=out/'visual.csv'
        postprocess(control,Path('/kaggle/working/visual_candidate_cache'),test,graph,out/'visual_receipt.json')
    else:graph=control/'submission.csv'
    if cfg['mode'] in ('dense','weighted'):
        from weighted_submission_finish import main as finish
        finish(package,test,control,config_name,graph,'joint_submission')
    else:
        shapes={p.stem:tuple(zarr.open_group(str(p),mode='r')['0'].shape) for p in sorted(test.glob('*.zarr'))}
        target=Path('/kaggle/working/submission.csv');repair=export_control(graph,target,shapes);groups=read_and_validate(target,shapes)
        (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,shapes=shapes,csv_sha256=sha(target),export_repair=repair,counts={k:dict(nodes=len(n),edges=len(e)) for k,(n,e) in groups.items()},annotations_read=False,cached_test_predictions=False,leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(*map(Path,sys.argv[1:4]),*(sys.argv[4:]))
