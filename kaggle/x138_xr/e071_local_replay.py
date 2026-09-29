"""Replay exact c3 postprocessing locally using common pre-ILP neural caches.

No leaderboard operations. Raw ILP solves run on CPU, DeepCenter on local CUDA.
The first remote candidate is the parity control, not a training/score proxy.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/E071'
OUT = ROOT / 'outputs/e071/local'
CACHE = ROOT / 'outputs/e071/shared/edge_cache'
BASE = ROOT / 'kaggle/x138_xr/kernels/k_c3/notebook.ipynb'
BASE_SHA = '052f47600befa2afc0cac6892a1f14aa0381dc786b5c322b0dd51752070b598b'
ARMS = {'C3_D04_R094': (.4,.94), 'C3_R094': (1.2,.94), 'C3_D04': (.4,.965),
        'C3_D08_R094': (.8,.94), 'C3_D04_R092': (.4,.92), 'C3_CONTROL': (1.2,.965)}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n');tmp.replace(p)


def pipeline():
    assert sha(BASE)==BASE_SHA
    nb=json.loads(BASE.read_text(encoding='utf8'));ns={'__name__':'e071_c3_replay'}
    # Reset only the task's environment knobs, so an earlier experiment cannot leak in.
    for key in list(os.environ):
        if key.startswith('BIOHUB_'): del os.environ[key]
    exec(compile(''.join(nb['cells'][0]['source']),str(BASE)+':0','exec'),ns)
    os.environ['BIOHUB_DEEPCENTER_CHECKPOINT']=str(ROOT/'artifacts/e071_runtime/deepcenter/best.pt')
    os.environ['BIOHUB_CACHE_DIR']=str(CACHE)
    assert sha(os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'])=='8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0'
    exec(compile(''.join(nb['cells'][2]['source']),str(BASE)+':2','exec'),ns)
    ns['TEST_DIR']=ROOT/'artifacts/e071_runtime/competition/test'
    ns['WORKING_DIR']=OUT
    ns['test_stems']=sorted(p.stem for p in CACHE.glob('*.npz'))
    ns['predict_seconds']=0
    tree=ast.parse(''.join(nb['cells'][5]['source']))
    assert ast.unparse(tree.body[-1])=="write_test_submission('base')"
    tree.body.pop()
    exec(compile(tree,str(BASE)+':5','exec'),ns)
    assert ns['DEEPCENTER_VETO_DETECTOR']['device'].type=='cuda'
    # Reuse exactly the same heatmap (including TTA) between parameter variants.
    original=ns['deepcenter_heatmap_for_frame']
    hmroot=OUT/'heatmaps';hmroot.mkdir(parents=True,exist_ok=True)
    def cached(dataset,t,detector_bundle,frame_cache,heatmap_cache):
        import numpy as np
        target=hmroot/dataset/f'{int(t):04d}.npy';target.parent.mkdir(exist_ok=True)
        if target.exists(): return np.load(target,allow_pickle=False)
        heatmap=original(dataset,t,detector_bundle,frame_cache,heatmap_cache)
        assert heatmap is not None and np.isfinite(heatmap).all()
        np.save(target,heatmap,allow_pickle=False)
        return heatmap
    # The original function's TTA marker is stored on its global name.
    ns['deepcenter_heatmap_for_frame']=cached
    return ns


def solve(stem,cost):
    import numpy as np
    import polars as pl
    import tracksdata as td
    path=OUT/'raw'/f'd{cost:g}'/(stem+'.json')
    cachepath=CACHE/(stem+'.npz');cachehash=sha(cachepath)
    if path.exists():
        result=json.loads(path.read_text());assert result['cache_sha256']==cachehash
        return result
    # Use the build_graph function from the scored support code, without its imports/main.
    source=ROOT/'outputs/e071/shared/tracking_repo/scripts/predict_unet_transformer.py'
    if not source.exists(): source=ROOT/'artifacts/e012_detector_check/repo/scripts/predict_unet_transformer.py'
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_graph')
    ns=dict(np=np,pl=pl,td=td)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    a=np.load(cachepath);edges=[(int(s),int(t),float(p),float(d)) for s,t,p,d in a['admitted']]
    g=ns['build_graph'](a['coords'],edges)
    solver=td.solvers.ILPSolver(edge_weight=-1.*td.EdgeAttr('edge_prob'),appearance_weight=0.,
        disappearance_weight=2.,division_weight=cost,timeout=1200)
    start=time.monotonic();g=solver.solve(g)
    nodes={int(r['node_id']):{k:(int(r[k]) if k in ('node_id','t') else float(r[k])) for k in ('node_id','t','z','y','x')} for r in g.node_attrs().iter_rows(named=True)}
    outedges=[dict(source_id=int(r['source_id']),target_id=int(r['target_id']),edge_prob=float(r['edge_prob'])) for r in g.edge_attrs().iter_rows(named=True)]
    result=dict(stem=stem,cost=cost,cache_sha256=cachehash,nodes=list(nodes.values()),edges=outedges,seconds=time.monotonic()-start)
    save(path,result)
    print('LOCAL_ILP',stem,cost,len(nodes),len(outedges),round(result['seconds'],2),flush=True)
    return result


def run(names,stems=None):
    if stems:
        manifest=json.loads((RES/'visible_input_manifest.json').read_text())
        for r in manifest:
            if r['name'].split('/')[1].removesuffix('.zarr') in stems:
                path=ROOT/'artifacts/e071_runtime/competition'/r['name']
                assert path.exists() and path.stat().st_size==r['totalBytes'],r['name']
    else:
        assert (RES/'local_visible_complete.json').exists(),'Download of all visible inputs must finish first'
    ns=pipeline()
    if stems: ns['test_stems']=stems
    for name in names:
        cost,readmit=ARMS[name];folder=(OUT/'partial' if stems else OUT)/name;folder.mkdir(parents=True,exist_ok=True)
        target=folder/'submission.csv';tmp=folder/'submission.partial.csv'
        assert not (folder/'verified.json').exists(),f'{name} already complete; use saved output'
        ns['READMIT_MIN_SCORE']=readmit
        stats=[];rowid=0
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=ns['CSV_COLUMNS']);writer.writeheader()
            for stem in ns['test_stems']:
                raw=solve(stem,cost);nodes={r['node_id']:dict(r) for r in raw['nodes']};edges=[dict(e) for e in raw['edges']]
                t0=time.monotonic()
                nodes,edges,s=ns['filter_output_graph'](nodes,edges,dataset=stem,deepcenter_bundle=ns['DEEPCENTER_VETO_DETECTOR'])
                assert nodes and edges
                for nodeid in sorted(nodes):
                    n=nodes[nodeid];writer.writerow(dict(id=rowid,dataset=stem,row_type='node',node_id=nodeid,t=n['t'],
                        z=max(0,round(n['z'])),y=max(0,round(n['y'])),x=max(0,round(n['x'])),source_id=-1,target_id=-1));rowid+=1
                for e in edges:
                    writer.writerow(dict(id=rowid,dataset=stem,row_type='edge',node_id=-1,t=-1,z=-1,y=-1,x=-1,
                        source_id=e['source_id'],target_id=e['target_id']));rowid+=1
                stats.append(dict(dataset=stem,nodes=len(nodes),edges=len(edges),seconds=time.monotonic()-t0,stats=s))
                save(folder/'progress.json',stats)
                print('LOCAL_REPLAY',name,stem,len(nodes),len(edges),flush=True)
        tmp.replace(target)
        from finalize_e071 import read_and_validate,graph_signature
        shapes=json.loads((ROOT/'results/E054_CONTROL_completed.json').read_text())['result']['shapes']
        graphs=read_and_validate(target,{k:v for k,v in shapes.items() if k in ns['test_stems']})
        signatures={k:graph_signature(*v)[3] for k,v in graphs.items()}
        record=dict(candidate=name,stage='PARTIAL_LOCAL_ONLY' if stems else 'LOCAL_VERIFIED_ONLY',gpu=ns['torch'].cuda.get_device_name(0),
            csv_sha256=sha(target),division_cost=cost,readmit=readmit,rows=rowid,stats=stats,
            graph_signatures=signatures,base_sha256=BASE_SHA,leaderboard_submitted=False)
        if name=='C3_D04_R094':
            remote=read_and_validate(ROOT/'outputs/e071/C3_D04_R094/submission.csv',shapes)
            record['remote_parity']={k:signatures[k]==graph_signature(*remote[k])[3] for k in signatures}
            save(folder/'verified.json',record)
            assert all(record['remote_parity'].values()),'Laptop replay differs from Kaggle; inspect before other candidates'
        save(folder/'verified.json',record)
        print('LOCAL_VERIFIED',name,'rows',rowid,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['solve','run']);p.add_argument('--names',nargs='+',default=['C3_D04_R094'])
    p.add_argument('--costs',type=float,nargs='+',default=[.4,1.2,.8]);p.add_argument('--stems',nargs='+')
    p.add_argument('--wait-visible-minutes',type=float,default=0);args=p.parse_args()
    if args.action=='solve':
        for cost in args.costs:
            for path in sorted(CACHE.glob('*.npz')): solve(path.stem,cost)
    else:
        deadline=time.monotonic()+args.wait_visible_minutes*60
        while not args.stems and not (RES/'local_visible_complete.json').exists() and time.monotonic()<deadline:
            time.sleep(30)
        run(args.names,args.stems)
