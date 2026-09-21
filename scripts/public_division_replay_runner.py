"""CPU factorial replay of public repair thresholds and visual associations."""
import copy,csv,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tracksdata as td
from ensemble_evaluate_runner import one,sha
from association_cpu_runner import evaluation_dir
from biohub_lab.public_postprocess import load
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.calibration_export import export_control
from biohub_lab.evaluate import evaluate_csv,shapes_for
from biohub_lab.visual_candidates import decode_cache
from biohub_lab.visual_assignment import assign

def predict(package,rank):
    out=Path('/kaggle/working/public_division_replay')/f'shard{rank}';out.mkdir(exist_ok=True)
    maps=one('public_division_fields/result.json');mr=json.loads(maps.read_text());assert mr['status']=='complete'
    capture=one('visual_validation_capture/result.json').parent;cr=json.loads((capture/'result.json').read_text());assert cr['status']=='complete'
    cfg=json.loads((package/'baseline/e052_protocol.json').read_text());assert cr['videos']==cfg['videos'];names=cfg['videos'][rank::4]
    module=load((package/'baseline/harmonic_inference.py').read_text());train=module.COMP_DIR/'train';module.TEST_DIR=train
    data=evaluation_dir(train,names,out/'data');shapes=shapes_for(data)
    bundle={'cfg':SimpleNamespace(pool_factor=mr['pool_factor'])}
    arms={f'd{d}_r{r}':(d/100,r/10) for d in (25,20) for r in (60,55)}
    names_arms=[a+s for a in arms for s in ('','_visual')]
    files={a:(out/(a+'_raw.csv')).open('w',newline='') for a in names_arms}
    writers={a:csv.writer(f) for a,f in files.items()};counters={a:0 for a in names_arms};reports=[]
    for w in writers.values():w.writerow(COLUMNS)
    def emit(arm,name,nodes,edges):
        for k in sorted(nodes):
            n=nodes[k];writers[arm].writerow([counters[arm],name,'node',k,int(n['t']),*[max(0,int(round(n[x]))) for x in ('z','y','x')],-1,-1]);counters[arm]+=1
        for a,b in edges:
            writers[arm].writerow([counters[arm],name,'edge',-1,-1,-1,-1,-1,a,b]);counters[arm]+=1
    try:
        cache_root=capture.parent/'visual_candidate_cache';manifest=json.loads((capture/'capture_manifest.json').read_text())
        for name in names:
            record=next(r for r in mr['videos'] if r['video']==name);path=maps.parent/record['file'];assert sha(path)==record['sha256']
            heat=np.load(path,mmap_mode='r');assert len(heat)==shapes[name][0]
            def cached(dataset,t,*args):
                assert dataset==name;return heat[int(t)]
            module.deepcenter_heatmap_for_frame=cached
            paths=list((capture/'harmonic_control/tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'));assert len(paths)==1
            raw=td.graph.IndexedRXGraph.from_geff(paths[0]);raw=raw[0] if isinstance(raw,tuple) else raw
            nodes={int(r['node_id']):dict(node_id=int(r['node_id']),t=int(r['t']),**{a:float(r[a]) for a in ('z','y','x')}) for r in raw.node_attrs().iter_rows(named=True)}
            edges=[dict(source_id=int(r['source_id']),target_id=int(r['target_id']),edge_prob=None if r.get('edge_prob') is None else float(r['edge_prob'])) for r in raw.edge_attrs().iter_rows(named=True)]
            frames=[]
            for p in sorted((cache_root/name).glob('*.npz')):
                assert sha(p)==manifest['files'][str(p.relative_to(cache_root))]
                with np.load(p,allow_pickle=False) as z:frames.append({k:z[k] for k in z.files})
            assert len(frames)==shapes[name][0]-1
            for arm,(threshold,radius) in arms.items():
                module.DEEPCENTER_SAFE_DIV_THRESHOLD=threshold;module.MOTION_RELINK_TIGHT_UM=radius
                final,links,stats=module.filter_output_graph(copy.deepcopy(nodes),copy.deepcopy(edges),dataset=name,deepcenter_bundle=bundle)
                original=[(e['source_id'],e['target_id']) for e in links];emit(arm,name,final,original)
                # Production visual solver uses rounded exported positions.
                exported={k:{axis:int(n[axis]) if axis=='t' else min(shapes[name][i]-1,max(0,int(round(n[axis])))) for i,axis in enumerate(('t','z','y','x'))} for k,n in final.items()}
                visual,audit=decode_cache(nodes,exported,frames);improved,vr=assign(exported,original,visual);emit(arm+'_visual',name,exported,improved)
                reports.append(dict(video=name,arm=arm,postprocess=stats,visual=vr,mapping=audit))
                print('ENSEMBLE_PUBLIC_REPLAY',name,arm,flush=True)
            del heat
    finally:
        for f in files.values():f.close()
    (out/'reports.json').write_text(json.dumps(reports))

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/public_division_replay');out.mkdir(exist_ok=True)
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i)],env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')) for i in range(4)]
    codes=[p.wait() for p in jobs];assert all(c==0 for c in codes),codes
    cfg=json.loads((package/'baseline/e052_protocol.json').read_text());names=cfg['videos']
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,names,out/'data');shapes=shapes_for(data)
    names_arms=[f'd{d}_r{r}'+s for d in (25,20) for r in (60,55) for s in ('','_visual')];reports=[]
    for i in range(4):reports+=json.loads((out/f'shard{i}/reports.json').read_text())
    for arm in names_arms:
        with (out/(arm+'_raw.csv')).open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(COLUMNS);index=0
            for i in range(4):
                with (out/f'shard{i}'/(arm+'_raw.csv')).open(newline='') as inp:
                    rows=csv.reader(inp);assert next(rows)==COLUMNS
                    for row in rows:row[0]=index;writer.writerow(row);index+=1
    metrics={};hashes={}
    for arm in names_arms:
        export_control(out/(arm+'_raw.csv'),out/(arm+'.csv'),shapes);hashes[arm]=sha(out/(arm+'.csv'))
    # All predictions are frozen before annotation-based evaluation.
    (out/'frozen_predictions.json').write_text(json.dumps(hashes,indent=2))
    for arm in names_arms:
        m=evaluate_csv(out/(arm+'.csv'),data);(out/(arm+'_metrics.json')).write_text(json.dumps(m,indent=2))
        metrics[arm]=dict(m['summary'],**{k:sum(s[k] for s in m['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
        print('ENSEMBLE_PUBLIC_METRIC',arm,metrics[arm],flush=True)
    reproduced=abs(metrics['d25_r60']['score']-.9436828898822122)<1e-10 and abs(metrics['d25_r60_visual']['score']-.9502832669356378)<1e-10
    selected=max(names_arms,key=lambda a:metrics[a]['score'])
    result=dict(status='complete',config=cfg,metrics=metrics,reports=reports,control_reproduced=reproduced,selected=selected,delta_vs_visual=metrics[selected]['score']-.9502832669356378,seconds=time.monotonic()-start,scope='Eight extensively reused videos; exploratory, not independent validation',leaderboard_submitted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));assert reproduced,'Control replay differs: investigate before promotion'
if __name__=='__main__':
    if len(sys.argv)==3:predict(Path(sys.argv[1]),int(sys.argv[2]))
    else:main(Path(sys.argv[1]))
