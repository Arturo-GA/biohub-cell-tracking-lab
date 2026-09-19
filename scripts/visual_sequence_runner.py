"""E020: paired visual-only and second-order visual graph decoding on CPU."""
from pathlib import Path
import csv,time,traceback
import numpy as np
import torch,zarr
from association_cpu_runner import locate,read,save,sha,evaluation_dir
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import shapes_for,evaluate_csv
from biohub_lab.visual_sequence import refine

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/visual_sequence')):
    root.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    result=dict(experiment='E020',status='predicting',accelerator='none',leaderboard_submitted=False)
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
        reports={};cache={}
        for name in names:
            paths=list((control/'harmonic_control/tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'))
            assert len(paths)==1,(name,paths)
            graph=zarr.open_group(str(paths[0]),mode='r')
            pairs=np.asarray(graph['edges/ids'][:]);prob=np.asarray(graph['edges/props/edge_prob/values'][:])
            assert pairs.shape==(len(prob),2),(pairs.shape,prob.shape)
            node_ids=np.asarray(graph['nodes/ids'][:]); coords={a:np.asarray(graph['nodes/props/'+a+'/values'][:]) for a in ('t','z','y','x')}
            # GEFF import reindexes IDs; export rounds coordinates to integer voxels.
            buckets={}
            for i,k in enumerate(node_ids):
                key=tuple(max(0,int(round(float(coords[a][i])))) for a in ('t','z','y','x'))
                buckets.setdefault(key,[]).append(int(k))
            export_buckets={}
            for k,v in reference[name][0].items():export_buckets.setdefault(tuple(v[a] for a in ('t','z','y','x')),[]).append(k)
            mapping={}; matched=0; ambiguous=0
            for k,v in reference[name][0].items():
                key=tuple(v[a] for a in ('t','z','y','x'));matches=buckets.get(key,[])
                if len(matches)==1 and len(export_buckets[key])==1:
                    mapping[matches[0]]=k;matched+=1
                elif matches:ambiguous+=1
            cache[name]={(mapping[int(a)],mapping[int(b)]):float(p) for (a,b),p in zip(pairs,prob) if int(a) in mapping and int(b) in mapping}
            reports[name]=dict(raw_edges=len(pairs),raw_nodes=len(node_ids),aligned_reference_nodes=matched,ambiguous_reference_nodes=ambiguous,reference_nodes=len(reference[name][0]))
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
        save(root/'result.json',result);print('E020_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
