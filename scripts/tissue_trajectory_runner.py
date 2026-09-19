"""E017 CPU: tissue-motion and long-context trajectories from frozen E016 nodes."""
from pathlib import Path
import time,traceback
import numpy as np
import torch
from association_cpu_runner import locate,read,save,sha,arrays,write_predictions_csv,evaluation_dir
from biohub_lab.tissue_trajectory import track
from biohub_lab.evaluate import evaluate_csv

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/tissue_trajectory')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E017',status='predicting',accelerator='none',training_started=False,leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU only')
        old=locate(input_root,'identity_parent/result.json').parent;pins=read(package/'baseline/e017_pins.json')
        assert sha(old/'result.json')==pins['result_sha256']
        assert sha(old/'geometric.csv')==pins['geometric_csv_sha256']
        assert sha(old/'geometric_metrics.json')==pins['geometric_metrics_sha256']
        names=read(package/'baseline/event_graph_split.json')['split']['calibration'];reports={}
        for name in names:
            d=arrays(old/'geometric'/name/'prediction.npz')
            edges,reports[name]=track(d['coords'],d['edges'])
            folder=root/'candidate'/name;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'prediction.npz',coords=d['coords'],edges=edges,shape=d['shape'])
            save(root/'trajectory_reports.json',reports);print('TISSUE_TRAJECTORY_READY',name,reports[name]['starts'],flush=True)
        write_predictions_csv(old/'geometric',names,root/'reconstructed_control.csv')
        assert sha(root/'reconstructed_control.csv')==pins['geometric_csv_sha256']
        csv=root/'candidate.csv';write_predictions_csv(root/'candidate',names,csv)
        save(root/'frozen_predictions.json',dict(csv_sha256=sha(csv),calibration_annotations_read=False,videos=names))
        train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
        metric=evaluate_csv(csv,evaluation_dir(train,names,root/'calibration_data'));save(root/'candidate_metrics.json',metric)
        assert sha(csv)==read(root/'frozen_predictions.json')['csv_sha256']
        before=read(old/'geometric_metrics.json')['summary']
        result.update(status='complete',seconds=time.monotonic()-start,candidate=metric['summary'],control=before,
            score_delta=metric['summary']['score']-before['score'],scope='16 calibration videos; primary detections, continuations only; full Harmonic control pending separately')
        save(root/'result.json',result);print('TISSUE_TRAJECTORY_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
