"""E021: cached pretrained visual-only and second-order visual graph decoding on CPU."""
from pathlib import Path
import csv,time,traceback
import numpy as np
import torch,zarr
from association_cpu_runner import locate,read,save,sha,evaluation_dir
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import shapes_for,evaluate_csv
from biohub_lab.visual_sequence import refine

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/visual_replay')):
    root.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    result=dict(experiment='E021',status='predicting',accelerator='none',leaderboard_submitted=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU only')
        compare=locate(input_root,'calibration_compare/result.json').parent
        control=locate(input_root,'calibration_control/result.json').parent
        pins=read(package/'baseline/e018_pins.json')
        assert sha(compare/'harmonic_validated.csv')==pins['harmonic_csv_sha256']
        assert sha(compare/'harmonic_metrics.json')==pins['harmonic_metrics_sha256']
        names=read(package/'baseline/event_graph_split.json')['split']['calibration']
        train=next(p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists())
        data=evaluation_dir(train,names,root/'calibration_data');shapes=shapes_for(data)
        reference=read_and_validate(compare/'harmonic_validated.csv',shapes)
        from biohub_lab.visual_replay import load_head,replay
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        source=control/'harmonic_control/tracking_repo/src/biohub_tracking/models/simple_node_transformer.py'
        weights=control/'harmonic_control/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth'
        replay_pins=read(package/'baseline/e021_pins.json')
        assert sha(source)==replay_pins['source_sha256']
        assert sha(weights)==replay_pins['checkpoint_sha256']
        model=load_head(source,weights)
        cache_root=locate(input_root,'event_graph_experiment/result.json').parent/'videos'
        reports={};cache={}
        for name in names:
            folder=cache_root/name;receipt=read(folder/'generation.json')
            for filename in ('graph.npz','features.npz'):assert sha(folder/filename)==receipt['files'][filename]
            with np.load(folder/'graph.npz',allow_pickle=False) as f:graph={k:f[k] for k in f.files}
            with np.load(folder/'features.npz',allow_pickle=False) as f:features=f['visual']
            cache[name],reports[name]=replay(reference[name][0],graph,features,model)
            np.savez_compressed(root/(name+'_visual.npz'),edges=np.array(list(cache[name]),dtype=np.int64),prob=np.array(list(cache[name].values())))
            save(root/'cache_audit.json',reports)
            print('REPLAY_READY',name,reports[name],flush=True)
        save(root/'cache_audit.json',reports)
        frozen={}
        for arm,weight in [('visual',0.),('sequence',.75)]:
            target=root/(arm+'.csv');arm_reports={};index=0
            with target.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(COLUMNS)
                for name in names:
                    nodes,old=reference[name];edges,arm_reports[name]=refine(nodes,old,cache[name],weight)
                    for k in sorted(nodes):
                        v=nodes[k];writer.writerow([index,name,'node',k,*[v[a] for a in ('t','z','y','x')],-1,-1]);index+=1
                    for a,b in edges:writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,a,b]);index+=1
                    print('VISUAL_SEQUENCE',arm,name,arm_reports[name],flush=True)
            read_and_validate(target,shapes);save(root/(arm+'_reports.json'),arm_reports);frozen[arm]=sha(target)
        save(root/'frozen_predictions.json',dict(csv_sha256=frozen,annotations_used=False))
        baseline=read(compare/'harmonic_metrics.json');result['harmonic']=baseline['summary'];result['arms']={}
        for arm in frozen:
            metric=evaluate_csv(root/(arm+'.csv'),data);save(root/(arm+'_metrics.json'),metric)
            assert sha(root/(arm+'.csv'))==frozen[arm]
            result['arms'][arm]=dict(summary=metric['summary'],score_delta=metric['summary']['score']-baseline['summary']['score'])
        result.update(status='complete',seconds=time.monotonic()-start,scope='Reused calibration, conditional on public pretrained models; not independent validation')
        save(root/'result.json',result);print('E021_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))

