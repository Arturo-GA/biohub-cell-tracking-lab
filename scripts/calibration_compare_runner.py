"""Evaluate completed full Harmonic in CPU and compare three frozen candidates."""
import time,traceback
from pathlib import Path
import torch
from association_cpu_runner import locate,read,save,sha,evaluation_dir
from biohub_lab.calibration_export import export_control
from biohub_lab.evaluate import evaluate_csv,shapes_for

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/calibration_compare')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E017-paired',status='verifying',accelerator='none',leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU only')
        pins=read(package/'baseline/e017_compare_pins.json')
        control=locate(input_root,'calibration_control/result.json').parent
        tissue=locate(input_root,'tissue_trajectory/result.json').parent
        parent=locate(input_root,'identity_parent/result.json').parent
        for key,folder in [('control',control),('tissue',tissue),('parent',parent)]:
            assert sha(folder/'result.json')==pins[key+'_result_sha256']
            assert read(folder/'result.json')['status']=='complete'
        names=read(package/'baseline/event_graph_split.json')['split']['calibration']
        assert read(control/'result.json')['videos']==names
        source=control/'harmonic_control/submission.csv';assert sha(source)==pins['control_csv_sha256']
        train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
        data=evaluation_dir(train,names,root/'calibration_data');target=root/'harmonic_validated.csv'
        receipt=export_control(source,target,shapes_for(data));receipt.update(source_sha256=sha(source),validated_sha256=sha(target),annotations_read=False)
        save(root/'export_receipt.json',receipt)
        # Reuse previously scored frozen candidates; no repeated training or inference.
        metrics={}
        for label,folder,filename in [('tissue',tissue,'candidate_metrics.json'),('learned',parent,'learned_metrics.json'),('geometric',parent,'geometric_metrics.json')]:
            assert sha(folder/filename)==pins[label+'_metrics_sha256']
            m=read(folder/filename)
            assert sorted(r['dataset'] for r in m['samples'])==sorted(names)
            assert m['official_commit']=='075fc5f5a52d11077f9dc2b074644618f26939e2'
            metrics[label]=m
        metrics['harmonic']=evaluate_csv(target,data);save(root/'harmonic_metrics.json',metrics['harmonic'])
        assert sha(target)==receipt['validated_sha256']
        reference={r['dataset']:r for r in metrics['harmonic']['samples']};paired={}
        for label,m in metrics.items():
            if label=='harmonic':continue
            paired[label]=[dict(dataset=r['dataset'],adj_edge_delta=r['adj_edge_jaccard']-reference[r['dataset']]['adj_edge_jaccard'],
                tp_delta=r['edge_tp']-reference[r['dataset']]['edge_tp'],fp_delta=r['edge_fp']-reference[r['dataset']]['edge_fp'],
                fn_delta=r['edge_fn']-reference[r['dataset']]['edge_fn']) for r in m['samples']]
        save(root/'paired.json',paired)
        result.update(status='complete',seconds=time.monotonic()-start,summaries={k:m['summary'] for k,m in metrics.items()},
            score_delta_vs_harmonic={k:m['summary']['score']-metrics['harmonic']['summary']['score'] for k,m in metrics.items() if k!='harmonic'},
            evaluation_cohort_used=False,scope='Same 16 calibration videos; conditional development, not a leaderboard score')
        save(root/'result.json',result);print('PAIRED_CALIBRATION_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
