"""E015: cached calibration identity audit and annotation-free graph projection."""
import json,time,traceback
from pathlib import Path
import numpy as np
import torch
from association_cpu_runner import locate,sha,read,save,arrays,write_predictions_csv,evaluation_dir
from biohub_lab.detection_identity import project_detections,identity_audit
from biohub_lab.temporal_data import load_gt
from biohub_lab.evaluate import evaluate_csv


def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/detection_identity')):
    if torch.cuda.is_available():raise RuntimeError('CPU only')
    torch.set_num_threads(2);root.mkdir(exist_ok=True);start=time.monotonic()
    result=dict(experiment='E015',status='verifying',accelerator='none',training_started=False,leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        pins=read(package/'baseline/e015_identity_pins.json')
        split=read(package/'baseline/event_graph_split.json')['split'];names=split['calibration']
        old=locate(input_root,'association_cpu/result.json').parent
        cache=locate(input_root,'event_graph_experiment/frozen_inputs.json').parent
        if sha(old/'result.json')!=pins['e014_cpu_result_sha256'] or sha(cache/'frozen_inputs.json')!=pins['e013_frozen_sha256']:
            raise ValueError('Source run changed')
        if sha(old/'calibration_predictions_frozen.json')!=pins['calibration_frozen_sha256']:
            raise ValueError('Calibration CSV provenance changed')
        baseline=old/'calibration_continuations.csv'
        if sha(baseline)!=pins['baseline_csv_sha256']:raise ValueError('Baseline CSV changed')
        before=read(old/'calibration_continuations_metrics.json')
        if sha(old/'calibration_continuations_metrics.json')!=pins['baseline_metrics_sha256']:raise ValueError('Baseline metric changed')
        records=read(cache/'frozen_inputs.json')['records'];reports={}
        for name in names:
            graph_path=cache/'videos'/name/'graph.npz'
            if sha(graph_path)!=records[name]['files']['graph.npz']:raise ValueError('Graph changed')
            graph=arrays(graph_path);pred=arrays(old/'calibration/continuations'/name/'prediction.npz')
            if not np.array_equal(graph['coords'],pred['coords']):raise ValueError('Node identity mismatch')
            # Verify original prediction arrays reproduce the pinned original CSV later.
            edges,report=project_detections(pred['coords'],pred['edges'],graph['origin'],radius=3.)
            target=root/'projected'/name;target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/'prediction.npz',coords=pred['coords'],edges=edges,shape=pred['shape'])
            reports[name]=report;print('IDENTITY_PROJECTION_READY',name,flush=True)
        rebuilt=root/'original_reconstructed.csv'
        write_predictions_csv(old/'calibration/continuations',names,rebuilt)
        if sha(rebuilt)!=pins['baseline_csv_sha256']:raise ValueError('Cached arrays do not reproduce original CSV')
        candidate=root/'projected.csv';write_predictions_csv(root/'projected',names,candidate)
        save(root/'frozen_prediction.json',dict(csv_sha256=sha(candidate),annotations_read=False,
            configuration=dict(radius_um=3.,priority='Harmonic, component span, ID',assignment='global maximum edge votes'),videos=names))
        result['status']='auditing_and_evaluating';save(root/'result.json',result)
        train=None
        for comp in (input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'):
            if (comp/'train').exists():train=comp/'train';break
        if train is None:raise FileNotFoundError('Competition missing')
        for name in names:
            truth,gt_edges=load_gt(train/(name+'.geff'))
            for arm,folder in [('original',old/'calibration/continuations'),('projected',root/'projected')]:
                pred=arrays(folder/name/'prediction.npz');active=np.unique(pred['edges'])
                remap=np.full(len(pred['coords']),-1,np.int64);remap[active]=np.arange(len(active))
                reports[name][arm]=identity_audit(pred['coords'][active],remap[pred['edges']],truth,gt_edges)
            save(root/'identity_audit.json',reports);print('IDENTITY_AUDIT_READY',name,flush=True)
        metric=evaluate_csv(candidate,evaluation_dir(train,names,root/'calibration_data'))
        if sha(candidate)!=read(root/'frozen_prediction.json')['csv_sha256']:raise ValueError('Prediction changed')
        save(root/'projected_metrics.json',metric)
        result.update(status='complete',seconds=time.monotonic()-start,videos=len(names),
            original=before['summary'],projected=metric['summary'],
            calibration_score_delta=metric['summary']['score']-before['summary']['score'],
            evaluation_cohort_used=False,next_action='Inspect identity discrepancies and full-graph ablation before any training or test submission',
            scope='Calibration diagnostic, not an improved Harmonic baseline or leaderboard submission')
        save(root/'result.json',result);print('IDENTITY_RESULT',json.dumps(result),flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
