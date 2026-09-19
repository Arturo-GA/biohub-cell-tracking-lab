"""E016 CPU training and paired full-graph calibration with frozen cached features."""
import time,traceback
from pathlib import Path
import numpy as np
import torch
from association_cpu_runner import locate,sha,read,save,arrays,write_predictions_csv,evaluation_dir
from biohub_lab.association_rank import node_features,SCALE
from biohub_lab.identity_parent import parent_table,supervision,ParentModel,decode,geometric_logits
from biohub_lab.temporal_data import load_gt
from biohub_lab.evaluate import evaluate_csv

CONFIG=dict(seed=20260919,steps=3000,batch=128,learning_rate=3e-4,parents=12,radius_um=20.,threads=2)
FROZEN='1c8f7d3d1134db7ed3d80355a3a61eac8b99bf2dd9dec68fc50362d16b6fb5c0'

def load_video(cache,name,record):
    folder=cache/'videos'/name
    for filename in ['graph.npz','features.npz']:
        if sha(folder/filename)!=record['files'][filename]:raise ValueError('Source cache changed')
    graph=arrays(folder/'graph.npz');ids=np.flatnonzero(graph['origin']==0)
    if not len(ids):raise ValueError('No primary detections')
    graph={k:v[ids] if k in ['coords','origin','scores'] else v for k,v in graph.items()}
    coords=graph['coords']
    if not ((coords>=0).all() and (coords<graph['shape']).all()):raise ValueError('Out of bounds primary detection')
    with np.load(folder/'features.npz',allow_pickle=False) as f:nodes=node_features(graph,f['visual'][ids])
    return dict(coords=coords,shape=graph['shape'],nodes=torch.from_numpy(nodes),
        positions=torch.from_numpy(coords[:,1:].astype(np.float32)*SCALE/20.),
        parents=torch.from_numpy(parent_table(coords,CONFIG['parents'],CONFIG['radius_um'])))

def train(fit,root,steps=CONFIG['steps']):
    torch.manual_seed(CONFIG['seed']);rng=np.random.default_rng(CONFIG['seed'])
    model=ParentModel();optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps)
    eligible=[n for n,v in fit.items() if len(v['labels']['targets'])]
    if not eligible:raise ValueError('No fit examples')
    history=[]
    for step in range(1,steps+1):
        v=fit[eligible[int(rng.integers(len(eligible)))]];d=v['labels']
        rows=rng.integers(len(d['targets']),size=CONFIG['batch'])
        targets=torch.from_numpy(d['targets'][rows]);y=torch.from_numpy(d['labels'][rows]);mask=torch.from_numpy(d['masks'][rows])
        logits=model(v['nodes'],v['positions'],v['parents'],targets).masked_fill(~mask,-1e4)
        loss=torch.nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        if step%250==0 or step==steps:
            history.append(dict(step=step,loss=float(loss.detach())))
            save(root/'training.json',dict(history=history,calibration_annotations_read=False))
            torch.save(dict(state_dict=model.state_dict(),step=step,config=CONFIG),root/'last.pt')
            print('IDENTITY_PARENT_TRAIN',history[-1],flush=True)
    return model.eval()

def main(package,input_root=Path('/kaggle/input'),root=Path('/kaggle/working/identity_parent')):
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    result=dict(experiment='E016',status='preparing',accelerator='none',leaderboard_submitted=False,evaluation_cohort_used=False)
    save(root/'result.json',result)
    try:
        if torch.cuda.is_available():raise RuntimeError('CPU session required')
        torch.set_num_threads(CONFIG['threads']);torch.set_num_interop_threads(1)
        cache=locate(input_root,'event_graph_experiment/frozen_inputs.json').parent
        if sha(cache/'frozen_inputs.json')!=FROZEN:raise ValueError('Source manifest changed')
        records=read(cache/'frozen_inputs.json')['records'];split=read(package/'baseline/event_graph_split.json')['split']
        assert not set(split['fit'])&set(split['calibration'])
        train_dir=next((p/'train' for p in [input_root/'competitions/biohub-cell-tracking-during-development',input_root/'biohub-cell-tracking-during-development'] if (p/'train').exists()),None)
        if train_dir is None:raise FileNotFoundError('Competition not attached')
        fit={};counts={}
        save(root/'config.json',dict(config=CONFIG,fit=split['fit'],calibration=split['calibration'],frozen_sha256=FROZEN,
            control='Geometric global assignment on same primary detections, not full Harmonic',checkpoint_selection='Fixed final step; no calibration selection'))
        for name in split['fit']:
            v=load_video(cache,name,records[name]);truth,edges=load_gt(train_dir/(name+'.geff'))
            v['labels']=supervision(v['coords'],v['parents'].numpy(),truth,edges)
            counts[name]={k:v['labels'][k] for k in ['matched_nodes','null_examples']};counts[name]['examples']=len(v['labels']['targets'])
            fit[name]=v;print('IDENTITY_FIT_READY',name,counts[name],flush=True)
        save(root/'fit_counts.json',counts);result['status']='training';save(root/'result.json',result)
        model=train(fit,root);del fit
        result.update(status='predicting',checkpoint_sha256=sha(root/'last.pt'));save(root/'result.json',result)
        for name in split['calibration']:
            v=load_video(cache,name,records[name]);parts=[]
            with torch.inference_mode():
                for first in range(0,len(v['coords']),1024):
                    targets=torch.arange(first,min(first+1024,len(v['coords'])))
                    parts.append(model(v['nodes'],v['positions'],v['parents'],targets).numpy())
            learned=np.concatenate(parts);parents=v['parents'].numpy()
            for arm,logits in [('learned',learned),('geometric',geometric_logits(v['coords'],parents))]:
                out=root/arm/name;out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'prediction.npz',coords=v['coords'],shape=v['shape'],edges=decode(v['coords'],parents,logits))
            print('IDENTITY_CALIBRATION_READY',name,flush=True)
        hashes={}
        for arm in ['learned','geometric']:
            path=root/(arm+'.csv');write_predictions_csv(root/arm,split['calibration'],path);hashes[arm]=sha(path)
        save(root/'frozen_predictions.json',dict(csv_sha256=hashes,calibration_annotations_read=False,checkpoint_sha256=sha(root/'last.pt')))
        data=evaluation_dir(train_dir,split['calibration'],root/'calibration_data');summaries={}
        for arm in ['learned','geometric']:
            metric=evaluate_csv(root/(arm+'.csv'),data);save(root/(arm+'_metrics.json'),metric);summaries[arm]=metric['summary']
            assert sha(root/(arm+'.csv'))==hashes[arm]
        result.update(status='complete',seconds=time.monotonic()-start,steps=CONFIG['steps'],summaries=summaries,
            score_delta=summaries['learned']['score']-summaries['geometric']['score'],
            scope='Conditional calibration on 16 videos; primary-only continuation model; not full Harmonic comparison')
        save(root/'result.json',result);print('IDENTITY_PARENT_RESULT',result,flush=True)
    except Exception as error:
        result.update(status='failed',error=str(error));save(root/'result.json',result)
        (root/'error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
