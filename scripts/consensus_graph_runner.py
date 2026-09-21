"""E047 full graph complement evaluation, all voting and association on CPU."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from ensemble_evaluate_runner import one,sha,matched
from biohub_lab.temporal_detector import vote_map
from biohub_lab.detector_complements import peaks,standardized,refine
from biohub_lab.temporal_bridge import propose,augment
from biohub_lab.submission import COLUMNS
from biohub_lab.evaluate import evaluate_csv
from association_cpu_runner import evaluation_dir

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e047_protocol.json').read_text());out=Path('/kaggle/working/consensus_graph');out.mkdir(exist_ok=True)
    fp=one('consensus_fields/result.json');assert sha(fp)==json.loads((package/'baseline/e047_infer_pin.json').read_text())['sha256'];fm=json.loads(fp.read_text());assert fm['config']==cfg
    rp=one('dense_detector_inputs/result.json');assert sha(rp)==cfg['raw_sha256'];raw=json.loads(rp.read_text())
    gp=one('temporal_graph_inputs/result.json');assert sha(gp)==cfg['graph_sha256'];graphs=json.loads(gp.read_text())
    arms=['control','dense_bridge','consensus_bridge','consensus_refine','consensus_refine_bridge'];handles={a:(out/(a+'.csv')).open('w',newline='') for a in arms};writers={a:csv.writer(h) for a,h in handles.items()};counter={a:0 for a in arms}
    for w in writers.values():w.writerow(COLUMNS)
    reports=[];donor_records=[]
    try:
        for g in graphs['videos']:
            video=g['video'];r=next(v for v in raw['videos'] if v['video']==video);f=next(v for v in fm['videos'] if v['video']==video)
            for path,digest in [(gp.parent/g['graph'],g['graph_sha256']),(rp.parent/r['file'],r['sha256']),(fp.parent/f['logits'],f['logits_sha256']),(fp.parent/f['field'],f['field_sha256'])]:assert sha(path)==digest
            with np.load(gp.parent/g['graph']) as z:ids,coords,edges=z['ids'],z['coords'],z['edges']
            image=np.load(rp.parent/r['file'],mmap_mode='r');logits=np.load(fp.parent/f['logits'],mmap_mode='r');fields=np.load(fp.parent/f['field'],mmap_mode='r')
            donors={'dense':[],'consensus':[]};refined=coords.copy().astype(np.float64)
            for t in range(len(image)):
                v=np.asarray(logits[t],np.float32);score=vote_map(np.asarray(fields[t],np.float32),np.clip(np.asarray(image[t],np.float32),0,1));consensus=standardized(v)+standardized(np.log1p(score))
                dp=peaks(v,cfg['budget']);cp=peaks(consensus,cfg['budget'])
                for name,points in [('dense',dp),('consensus',cp)]:donors[name].append(np.c_[np.full(len(points),t),points*np.array([1,4,4])])
                ix=np.flatnonzero(coords[:,0]==t);anchors=coords[ix,1:]/[1,4,4]
                refined[ix,1:]=refine(anchors,cp)*[1,4,4]
            donors={k:np.concatenate(v).astype(np.int64) for k,v in donors.items()};refined=np.rint(refined).astype(np.int64)
            donorfile=out/(video+'_candidates.npz');np.savez_compressed(donorfile,**donors);donor_records.append(dict(video=video,file=donorfile.name,sha256=sha(donorfile)))
            proposals={k:propose(ids,coords,edges,d) for k,d in donors.items()};combo=propose(ids,refined,edges,donors['consensus'])
            results={'control':(ids,coords,edges,{}),'consensus_refine':(ids,refined,edges,dict(moved_nodes=int(np.any(refined!=coords,axis=1).sum())))}
            results['dense_bridge']=augment(ids,coords,edges,donors['dense'],proposals['dense'])
            results['consensus_bridge']=augment(ids,coords,edges,donors['consensus'],proposals['consensus'])
            results['consensus_refine_bridge']=augment(ids,refined,edges,donors['consensus'],combo)
            reports.append(dict(video=video,arms={a:r[3] for a,r in results.items()}))
            for a,(ns,cs,es,report) in results.items():
                for node,c in zip(ns,cs):writers[a].writerow([counter[a],video,'node',int(node),*map(int,c),-1,-1]);counter[a]+=1
                for source,target in es:writers[a].writerow([counter[a],video,'edge',-1,-1,-1,-1,-1,int(source),int(target)]);counter[a]+=1
            print('ENSEMBLE_GRAPH_VIDEO',video,reports[-1]['arms'],flush=True)
    finally:
        for h in handles.values():h.close()
    inputs=Path('/kaggle/input');train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists());evaluation=evaluation_dir(train,[g['video'] for g in graphs['videos']],out/'data')
    metrics={};hashes={};details={}
    for a in arms:
        path=out/(a+'.csv');hashes[a]=sha(path);metric=evaluate_csv(path,evaluation);assert sha(path)==hashes[a];details[a]=metric
        (out/(a+'_metrics.json')).write_text(json.dumps(metric,indent=2));metrics[a]=dict(metric['summary'],**{k:sum(r[k] for r in metric['samples']) for k in ['edge_tp','edge_fp','edge_fn']});print('ENSEMBLE_GRAPH_METRIC',a,metrics[a],flush=True)
    c=metrics['control'];assert abs(c['score']-.9007529595605399)<1e-10
    from biohub_official.metrics import summarise
    embryo_metrics={a:{e:summarise([r for r in details[a]['samples'] if r['dataset'].startswith(e)]) for e in ['44b6','6bba']} for a in arms}
    passed={a:metrics[a]['score']>=c['score']+.001 and metrics[a]['edge_tp']>c['edge_tp'] and metrics[a]['edge_fp']<=c['edge_fp'] and metrics[a]['division_tp']>=c['division_tp'] and metrics[a]['division_fp']<=c['division_fp'] for a in arms if a!='control'}
    passed={a:ok and all(embryo_metrics[a][e]['score']>=embryo_metrics['control'][e]['score'] for e in ['44b6','6bba']) for a,ok in passed.items()}
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,metrics=metrics,passed=passed,embryo_metrics=embryo_metrics,reports=reports,videos=donor_records,csv_sha256=hashes,seconds=time.monotonic()-start,leaderboard_submitted=False,scope='Exploratory full-graph reused calibration; selected512 after E045 diagnostic; not independent validation. Embryo gate evaluated from official per-video summaries.'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
