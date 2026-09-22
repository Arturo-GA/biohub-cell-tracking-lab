import csv,json,shutil,sys,time
from pathlib import Path
from joint_submission_runner import main as harmonic
from biohub_lab.appearance_swap import describe,refine
from biohub_lab.submission import read_and_validate,COLUMNS
from ensemble_evaluate_runner import sha

def main(package):
    start=time.monotonic();harmonic(package,'e065_submission.json')
    import zarr  # Harmonic installs the pinned offline wheel before this import.
    out=Path('/kaggle/working/appearance_swap_submission');out.mkdir(exist_ok=True);target=Path('/kaggle/working/submission.csv');original=out/'control.csv';shutil.copyfile(target,original)
    base=json.loads(Path('/kaggle/working/joint_submission/result.json').read_text());cfg=base['config'];shapes=base['shapes'];graphs=read_and_validate(original,shapes)
    test=next(p/'test' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'test').exists());reports=[]
    with target.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(COLUMNS);i=0
        for video,(nodes,edges) in graphs.items():
            image=zarr.open_group(str(test/(video+'.zarr')),mode='r')['0'];features=describe(nodes,image);es,report=refine(nodes,edges,features,**cfg['swap_options']);ns=nodes;reports.append(dict(video=video,**report))
            for k in sorted(ns):w.writerow([i,video,'node',k,*[ns[k][a] for a in ('t','z','y','x')],-1,-1]);i+=1
            for a,b in es:w.writerow([i,video,'edge',-1,-1,-1,-1,-1,a,b]);i+=1
            print('ENSEMBLE_APPEARANCE_SUBMISSION',reports[-1],flush=True)
    final=read_and_validate(target,shapes)
    assert all(final[v][0]==graphs[v][0] and len(final[v][1])==len(graphs[v][1]) for v in graphs)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,shapes=shapes,reports=reports,csv_sha256=sha(target),control_sha256=sha(original),seconds=time.monotonic()-start,annotations_read=False,cached_test_predictions=False,leaderboard_submitted=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
