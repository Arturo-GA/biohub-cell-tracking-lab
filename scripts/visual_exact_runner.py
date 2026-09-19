"""E022 CPU: exact visual alternatives plus sequence optimization."""
from pathlib import Path
import csv,time,traceback
import numpy as np
import torch,tracksdata as td
from association_cpu_runner import locate,read,save,sha,evaluation_dir
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import shapes_for,evaluate_csv
from biohub_lab.visual_candidates import decode_cache
from biohub_lab.visual_sequence import refine

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/visual_exact')):
    root.mkdir(exist_ok=True);start=time.monotonic()
    result=dict(experiment='E022-CPU',status='predicting',accelerator='none',leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU only')
        control=locate(input_root,'visual_capture/result.json').parent
        cr=read(control/'result.json');assert cr['status']=='complete'
        names=read(package/'baseline/e022_capture.json')['videos'];assert cr['videos']==names
        train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
        data=evaluation_dir(train,names,root/'data');shapes=shapes_for(data)
        source=control/'harmonic_control/submission.csv';assert sha(source)==cr['csv_sha256']
        export=export_control(source,root/'harmonic.csv',shapes);save(root/'export_receipt.json',export)
        reference=read_and_validate(root/'harmonic.csv',shapes)
        cache_root=control.parent/'visual_candidate_cache';manifest=read(control/'capture_manifest.json')
        cache={};reports={}
        for name in names:
            files=list((control/'harmonic_control/tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'));assert len(files)==1
            raw=td.graph.IndexedRXGraph.from_geff(files[0]);raw=raw[0] if isinstance(raw,tuple) else raw
            nodes={int(row['node_id']):{a:float(row[a]) for a in ('t','z','y','x')} for row in raw.node_attrs().iter_rows(named=True)}
            frames=[]
            for p in sorted((cache_root/name).glob('*.npz')):
                assert sha(p)==manifest['files'][str(p.relative_to(cache_root))]
                with np.load(p,allow_pickle=False) as values:frames.append({k:values[k] for k in values.files})
            assert len(frames)==shapes[name][0]-1
            cache[name],reports[name]=decode_cache(nodes,reference[name][0],frames)
            np.savez_compressed(root/(name+'_visual.npz'),edges=np.array(list(cache[name]),np.int64),prob=np.array(list(cache[name].values())))
        save(root/'cache_audit.json',reports)
        frozen={'harmonic':sha(root/'harmonic.csv')}
        for arm,weight in [('visual',0.),('sequence',.75)]:
            target=root/(arm+'.csv');arm_reports={};index=0
            with target.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(COLUMNS)
                for name in names:
                    nodes,original=reference[name];edges,arm_reports[name]=refine(nodes,original,cache[name],weight)
                    for k in sorted(nodes):
                        v=nodes[k];writer.writerow([index,name,'node',k,*[v[a] for a in ('t','z','y','x')],-1,-1]);index+=1
                    for a,b in edges:writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,a,b]);index+=1
                    print('EXACT_READY',arm,name,arm_reports[name],flush=True)
            read_and_validate(target,shapes);frozen[arm]=sha(target);save(root/(arm+'_reports.json'),arm_reports)
        save(root/'frozen_predictions.json',dict(csv_sha256=frozen,annotations_used=False,videos=names))
        summaries={}
        for arm in frozen:
            metric=evaluate_csv(root/(arm+'.csv'),data);save(root/(arm+'_metrics.json'),metric)
            assert sha(root/(arm+'.csv'))==frozen[arm];summaries[arm]=metric['summary']
        result.update(status='complete',seconds=time.monotonic()-start,summaries=summaries,
            deltas={k:v['score']-summaries['harmonic']['score'] for k,v in summaries.items() if k!='harmonic'},
            scope='Two fixed calibration videos; smoke only, not validation or submission evidence')
        save(root/'result.json',result);print('EXACT_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
