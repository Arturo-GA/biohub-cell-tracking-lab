"""Official-matcher residual and unique-correct-edge audit. CPU, diagnostic only."""
import json,time,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import tracksdata as td
import polars as pl
from association_cpu_runner import evaluation_dir
from biohub_lab.submission import read_and_validate
from biohub_lab.evaluate import shapes_for
from biohub_official.metrics import _evaluate,_evaluate_matched_graph

def roots(relative):
    base=Path('/kaggle/input')
    return list(dict.fromkeys(p for depth in [1,2,3] for p in base.glob('*/'*depth+relative)))

def match(nodes,edges,truth,expected):
    k=td.DEFAULT_ATTR_KEYS;graph=td.graph.InMemoryGraph()
    for axis in ('z','y','x'):graph.add_node_attr_key(axis,pl.Float64,0.)
    keys=sorted(nodes);ids=graph.bulk_add_nodes([dict(t=nodes[n]['t'],**{a:float(nodes[n][a]) for a in ('z','y','x')}) for n in keys]);mapping=dict(zip(keys,ids));reverse=dict(zip(ids,keys))
    graph.bulk_add_edges([dict(source_id=mapping[a],target_id=mapping[b]) for a,b in edges])
    _evaluate(graph,truth,'jaccard',(1.625,.40625,.40625),7.)
    mapping={reverse[r[k.NODE_ID]]:r[k.MATCHED_NODE_ID] for r in graph.node_attrs(attr_keys=[k.NODE_ID,k.MATCHED_NODE_ID]).iter_rows(named=True)}
    inverse=defaultdict(list)
    for p,g in mapping.items():
        if g is not None and g!=-1:inverse[int(g)].append(p)
    tp=set();fp=0
    for row in _evaluate_matched_graph(graph,truth).iter_rows(named=True):
        a,b=reverse[row[k.EDGE_SOURCE]],reverse[row[k.EDGE_TARGET]]
        if row[k.MATCHED_EDGE_MASK]:tp.add((int(mapping[a]),int(mapping[b])))
        elif row['pred_valid']:fp+=1
    assert len(tp)==expected['edge_tp'] and fp==expected['edge_fp']
    assert truth.num_edges()-len(tp)==expected['edge_fn']
    return tp,inverse

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/residual_audit');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e058_protocol.json').read_text());views={}
    for path in roots('visual_validation/result.json'):
        r=json.loads(path.read_text());assert r['status']=='complete'
        label='synthetic' if r.get('experiment')=='E056' else 'original'
        views[label]=path.parent
    assert set(views)=={'original','synthetic'}
    fixed=roots('synthetic_fixed_graph/result.json');assert len(fixed)==1;fixed=fixed[0].parent
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    data=evaluation_dir(train,cfg['videos'],out/'data');shapes=shapes_for(data)
    paths={'harmonic':(views['original'],'harmonic'),'visual':(views['original'],'visual'),'synthetic_visual':(views['synthetic'],'visual'),**{a:(fixed,a) for a in ['specialist','balanced','uncertainty']}}
    graphs={a:read_and_validate(root/(stem+'.csv'),shapes) for a,(root,stem) in paths.items()}
    metrics={a:{r['dataset']:r for r in json.loads((root/(stem+'_metrics.json')).read_text())['samples']} for a,(root,stem) in paths.items()}
    reports=[];residuals=[]
    for name in cfg['videos']:
        truth=td.graph.IndexedRXGraph.from_geff(str(data/(name+'.geff')))[0];k=td.DEFAULT_ATTR_KEYS
        gt={(int(r[k.EDGE_SOURCE]),int(r[k.EDGE_TARGET])) for r in truth.edge_attrs(attr_keys=[k.EDGE_SOURCE,k.EDGE_TARGET]).iter_rows(named=True)};degree=Counter(a for a,b in gt)
        recovered={a:match(*graphs[a][name],truth,metrics[a][name]) for a in graphs};base=recovered['visual'][0]
        for arm,(tp,_) in recovered.items():
            reports.append(dict(video=name,arm=arm,tp=len(tp),rescued=sorted(tp-base),lost=sorted(base-tp)))
        nodes,edges=graphs['visual'][name];inverse=recovered['visual'][1]
        with np.load(views['original']/(name+'_visual.npz')) as z:probs={tuple(map(int,e)):float(p) for e,p in zip(z['edges'],z['prob'])}
        by_target=defaultdict(list)
        for (a,b),p in probs.items():by_target[b].append((a,p))
        pred_degree=Counter(a for a,b in edges)
        for a,b in sorted(gt-base):
            src,tgt=inverse.get(a,[]),inverse.get(b,[]);row=dict(video=name,truth_edge=[a,b],division_edge=degree[a]==2)
            if not src or not tgt:row['category']='missing_both' if not src and not tgt else 'missing_source' if not src else 'missing_target'
            else:
                available=[(probs[x,y],x,y) for x in src for y in tgt if (x,y) in probs]
                if not available:row['category']='outside_cached_candidates'
                else:
                    p,x,y=max(available);values=sorted((q for _,q in by_target[y]),reverse=True)
                    row.update(category='in_cached_candidates',predicted_edge=[x,y],probability=p,column_rank=1+sum(q>p for q in values),column_margin=values[0]-values[1] if len(values)>1 else 1.,locked_source_division=pred_degree[x]>1,distance_um=float(np.linalg.norm((np.array([nodes[y][ax] for ax in ('z','y','x')])-np.array([nodes[x][ax] for ax in ('z','y','x')]))*np.array([1.625,.40625,.40625]))))
            residuals.append(row)
        print('ENSEMBLE_RESIDUAL',name,dict(Counter(r['category'] for r in residuals if r['video']==name)),flush=True)
    totals={a:dict(rescued=sum(len(r['rescued']) for r in reports if r['arm']==a),lost=sum(len(r['lost']) for r in reports if r['arm']==a)) for a in paths}
    result=dict(status='complete',seconds=time.monotonic()-start,config=cfg,complementarity=totals,by_video=reports,residual_categories=dict(Counter(r['category'] for r in residuals)),residuals=residuals,scope='Ground-truth diagnostic on reused eight videos. Oracle edge unions are not deployable scores; no submission generated.',leaderboard_submitted=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('ENSEMBLE_RESIDUAL_COMPLETE',totals,result['residual_categories'],flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
