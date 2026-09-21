"""CPU-only joint evaluation of public repairs, visual links and dense bridges."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from ensemble_evaluate_runner import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.temporal_detector import vote_map
from biohub_lab.detector_complements import peaks,standardized
from biohub_lab.weighted_bridge import combine
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.evaluate import evaluate_csv,shapes_for

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/complement_joint');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e053_protocol.json').read_text());fp=one('complement_fields_reserved/result.json');fm=json.loads(fp.read_text());assert fm['status']=='complete' and fm['config']==cfg
    rp=one('public_division_replay/result.json');rm=json.loads(rp.read_text());assert rm['status']=='complete' and rm['control_reproduced']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data)
    frozen=json.loads((rp.parent/'frozen_predictions.json').read_text());graphs={}
    for arm in rm['metrics']:
        assert sha(rp.parent/(arm+'.csv'))==frozen[arm];graphs[arm]=read_and_validate(rp.parent/(arm+'.csv'),shapes)
    arms={base+'_'+mode:(base,mode) for base in graphs for mode in ('dense','weighted')}
    handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(f) for a,f in handles.items()};index={a:0 for a in arms};reports=[];donor_records=[]
    for w in writers.values():w.writerow(COLUMNS)
    try:
        for video in cfg['videos']:
            r=next(r for r in fm['videos'] if r['video']==video);values={}
            for key,entry in r['files'].items():
                p=fp.parent/entry['file'];assert sha(p)==entry['sha256'];values[key]=np.load(p,mmap_mode='r')
            donors=[];consensus=[]
            for t in range(len(values['image'])):
                logits=values['logits'][t].astype(np.float32);field=values['field'][t].astype(np.float32);image=np.clip(values['image'][t].astype(np.float32),0,1)
                dn=peaks(logits,512);cn=peaks(standardized(logits)+standardized(np.log1p(vote_map(field,image))),512)
                donors.append(np.c_[np.full(len(dn),t),dn*[1,4,4]]);consensus.append(np.c_[np.full(len(cn),t),cn*[1,4,4]])
            dn=np.concatenate(donors).astype(np.int64);cn=np.concatenate(consensus).astype(np.int64)
            path=out/(video+'_donors.npz');np.savez_compressed(path,dense=dn,consensus=cn);donor_records.append(dict(video=video,file=path.name,sha256=sha(path)))
            for arm,(base,mode) in arms.items():
                nodes,edges=graphs[base][video];ids=np.array(sorted(nodes));coords=np.array([[nodes[int(k)][axis] for axis in ('t','z','y','x')] for k in ids]);edges=np.asarray(edges,np.int64).reshape(-1,2)
                ns,cs,es,report=combine(ids,coords,edges,dn,cn,mode)
                for k,c in zip(ns,cs):writers[arm].writerow([index[arm],video,'node',int(k),*map(int,c),-1,-1]);index[arm]+=1
                for a,b in es:writers[arm].writerow([index[arm],video,'edge',-1,-1,-1,-1,-1,int(a),int(b)]);index[arm]+=1
                reports.append(dict(video=video,arm=arm,**report))
            print('ENSEMBLE_JOINT_COMPLEMENTS',video,flush=True)
    finally:
        for h in handles.values():h.close()
    hashes={a:sha(out/(a+'.csv')) for a in arms};(out/'frozen_predictions.json').write_text(json.dumps(hashes,indent=2))
    metrics=dict(rm['metrics'])
    for arm in arms:
        read_and_validate(out/(arm+'.csv'),shapes);m=evaluate_csv(out/(arm+'.csv'),data);assert sha(out/(arm+'.csv'))==hashes[arm]
        (out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2));metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        print('ENSEMBLE_JOINT_METRIC',arm,metrics[arm],flush=True)
    selected=max(metrics,key=lambda a:(metrics[a]['score'],-metrics[a]['edge_fp'],-len(a)))
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,selected=selected,reports=reports,videos=donor_records,seconds=time.monotonic()-start,delta_vs_prior_visual=metrics[selected]['score']-.9502832669356378,leaderboard_submitted=False,scope='Eight reused videos, exploratory factorial comparison, no independent holdout'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
