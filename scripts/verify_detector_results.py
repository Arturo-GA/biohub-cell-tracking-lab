"""Audit completed E008/E009 outputs and exact-frame mitosis opportunities.

No inference is rerun, no thresholds are selected, and no predictions change.
The local metric check recomputes aggregation, not the official graph matching.
"""
import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
from biohub_lab.detector_proposals import install_pipeline_hook, SCALE
from biohub_lab.patch import patch_source
from biohub_lab.submission import read_and_validate

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_aggregator():
    """Execute the pinned, dependency-free aggregation functions verbatim."""
    tree=ast.parse((ROOT/'src/biohub_official/metrics.py').read_text(encoding='utf8'))
    wanted=[]
    for node in tree.body:
        if isinstance(node,ast.AnnAssign) and node.target.id in ('COUNT_COLUMNS','SCORE_DIVISION_WEIGHT'):
            wanted.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in ('_jaccard','summarise'):
            wanted.append(node)
    namespace={'warnings':warnings}
    exec(compile(ast.Module(body=wanted,type_ignores=[]),'<pinned-metric-aggregation>','exec'),namespace)
    return namespace['summarise']


def verify(number, control):
    root=ROOT/'outputs'/(number.lower()+'_v2')
    result=read(root/'detector_experiment/result.json')
    status=read(root/'status.json');launch=read(ROOT/'results'/(number+'_launch.json'))
    assert status['status']=='COMPLETE' and result['status']=='complete' and launch['versionNumber']==2
    notebook,=list((root/'source').glob('*.ipynb'))
    source='\n'.join(''.join(c['source']) for c in read(notebook)['cells'] if c['cell_type']=='code')
    assignment=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.startswith('src/') and name.endswith('.py'):
                assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),name
        baseline=archive.read('baseline/harmonic_inference.py').decode()
    expected_run=install_pipeline_hook(patch_source(baseline,candidate=False)).replace(
        '/kaggle/working','/kaggle/working/detector_experiment/tracking')
    run=root/'detector_experiment/tracking/run.py'
    assert sha(run)==result['run_source_sha256']==hashlib.sha256(expected_run.encode()).hexdigest()
    shapes={};proposals={}
    for name,receipt in result['proposal_receipts'].items():
        path=root/'detector_experiment/proposals'/(name+'.npz')
        assert sha(path)==receipt['sha256']
        with np.load(path,allow_pickle=False) as arrays:
            points,scores,shape=arrays['coords'],arrays['scores'],arrays['shape']
        assert points.shape==(receipt['proposals'],4) and len(scores)==len(points)
        assert np.isfinite(points).all() and np.isfinite(scores).all()
        assert np.all(points>=0) and np.all(points<=shape-1)
        assert np.all(points[:,0]==points[:,0].astype(int))
        assert [int((points[:,0]==t).sum()) for t in range(shape[0])]==receipt['frame_counts']
        shapes[name]=tuple(map(int,shape));proposals[name]=points
    csv_path=root/'detector_experiment/tracking/submission.csv'
    assert sha(csv_path)==result['csv_sha256']
    parsed=read_and_validate(csv_path,shapes)
    assert set(parsed)==set(result['datasets'])==set(control['datasets'])
    counts={name:dict(nodes=len(nodes),edges=len(edges)) for name,(nodes,edges) in parsed.items()}
    assert counts==result['final_counts']
    injections=[]
    for file in (root/'detector_experiment/proposals').glob('injection_*.jsonl'):
        injections.extend(json.loads(line) for line in file.read_text().splitlines())
    expected={(name,t) for name,shape in shapes.items() for t in range(shape[0])}
    assert {(r['dataset'],r['t']) for r in injections}==expected and len(injections)==len(expected)
    for name in shapes:
        rows=[r for r in injections if r['dataset']==name]
        assert {key:sum(r[key] for r in rows) for key in ('baseline','proposals','added')}==result['injection_summary'][name]
        assert all(r['proposals']==result['proposal_receipts'][name]['frame_counts'][r['t']] for r in rows)
    summary=official_aggregator()(result['metrics']['samples'])
    for key,value in summary.items(): np.testing.assert_allclose(value,result['metrics']['summary'][key],atol=1e-14)
    for row in result['metrics']['samples']:
        assert row['num_pred_nodes']==len(parsed[row['dataset']][0])
    assert result['control']==control['arms']['control']['metrics']['summary']
    for key,delta in result['delta'].items(): np.testing.assert_allclose(summary[key]-result['control'][key],delta,atol=1e-14)
    integrity=read(root/'detector_experiment/tracking/bidirectional_production_runtime_integrity.json')
    assert integrity['verified_before_dynamic_source_patch'] and not integrity['ground_truth_accessed']
    assert integrity['checkpoint_sha256']=={
        'primary':'12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771',
        'secondary':'9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f',
        'deepcenter':'8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0'}
    total_nodes=sum(len(nodes) for nodes,_ in parsed.values())
    total_edges=sum(len(edges) for _,edges in parsed.values())
    forks=sum(sum(n==2 for n in Counter(s for s,t in edges).values()) for _,edges in parsed.values())
    completed={k:v for k,v in result.items() if k!='proposal_receipts'}
    completed.update(experiment=number,version=2,kernel=launch['url'],checked_at_utc=status['checked_at_utc'],
        payload_sha256=launch['payload_sha256'],quality_status='below_control_no_additional_official_division',
        decision='do_not_submit_this_version',totals=dict(nodes=total_nodes,edges=total_edges,predicted_forks=forks,
            proposals=sum(len(p) for p in proposals.values()),added_before_ilp=sum(r['added'] for r in injections),
            edge_tp=sum(r['edge_tp'] for r in result['metrics']['samples']),
            edge_fp=sum(r['edge_fp'] for r in result['metrics']['samples']),
            edge_fn=sum(r['edge_fn'] for r in result['metrics']['samples'])),
        verification=dict(source_payload=True,run_source=True,checkpoint_integrity_receipt=True,
            final_csv=True,proposals=True,injection_frames=len(injections),metric_aggregation=True,
            official_graph_matching_rerun_locally=False))
    (ROOT/'results'/(number+'_completed.json')).write_text(json.dumps(completed,indent=2)+'\n')
    print(number,'score',summary['score'],'delta',result['delta']['score'],'totals',completed['totals'])
    return parsed,proposals,shapes


def geometry(points, gt, edges=()):
    nearest=[];distances=[];within=[]
    for p in gt:
        indices=np.flatnonzero(points[:,0]==p[0])
        d=np.linalg.norm((points[indices,1:]-p[1:])*SCALE,axis=1)
        near=int(indices[np.argmin(d)]) if len(d) else None
        nearest.append(near);distances.append(float(d.min()) if len(d) else None)
        within.append(indices[d<=7.].tolist())
    children=defaultdict(list);parents={}
    for s,t in edges:children[int(s)].append(int(t));parents[int(t)]=int(s)
    distinct=any(a!=b for a in within[1] for b in within[2])
    forks=[m for m in within[0] if any(a!=b and a in children[m] and b in children[m]
                                    for a in within[1] for b in within[2])]
    return dict(nearest_indices=nearest,nearest_distances_um=distances,
        daughters_share_nearest=nearest[1]==nearest[2],distinct_daughter_candidates_within_7um=distinct,
        nearby_fork_mothers=forks,nearest_mother_children=children[nearest[0]],
        nearest_daughter_parents=[parents.get(n) for n in nearest[1:]])


def graph_arrays(nodes,edges):
    ids=sorted(nodes);lookup={v:i for i,v in enumerate(ids)}
    points=np.array([[nodes[i][k] for k in ('t','z','y','x')] for i in ids])
    return points,[(lookup[s],lookup[t]) for s,t in edges]


def main():
    control=read(ROOT/'outputs/diagnostic/run_receipt.json')
    control_csv=ROOT/'outputs/diagnostic/biohub_control/submission.csv'
    assert sha(control_csv)==control['arms']['control']['sha256']
    models={};proposals={};shapes=None
    for number in ('E008','E009'):
        models[number],proposals[number],shapes=verify(number,control)
    models['control']=read_and_validate(control_csv,shapes)
    arrays={arm:{name:graph_arrays(*graph) for name,graph in data.items()} for arm,data in models.items()}
    events=[];gt_hashes={}
    for name in shapes:
        path=ROOT/'outputs/mitosis_audit/temporal_cache'/name/'graph.npz'
        gt_hashes[name]=sha(path);receipt=read(path.parent/'receipt.json')
        with np.load(path,allow_pickle=False) as data:gt={k:data[k] for k in data.files}
        digest=hashlib.sha256(json.dumps(receipt['config'],sort_keys=True).encode()+
            gt['coords'][:receipt['gt_nodes']].tobytes()+gt['edges'].tobytes()).hexdigest()
        assert digest==receipt['config_hash']
        for ref in np.flatnonzero(gt['triple_labels']==1):
            truth=gt['coords'][gt['triples'][ref]]
            events.append(dict(dataset=name,mother_frame=int(truth[0,0]),gt_coords=truth.tolist(),
                final={arm:geometry(points,truth,edges) for arm,graphs in arrays.items() for key,(points,edges) in graphs.items() if key==name},
                proposals={arm:geometry(data[name],truth) for arm,data in proposals.items()}))
    assert len(events)==7
    audit=dict(events=events,gt_graph_sha256=gt_hashes,prepared_GT_checksums_verified=True,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),predictions_changed=False,
        scope='Exact-frame geometric diagnostic at 7um, not official matching or a score oracle. '
              'Proposal-to-final absence cannot distinguish NMS/budget rejection, ILP selection and postprocessing. '
              'These seven events were inspected after results; not independent validation.')
    # Detailed annotation coordinates remain local in the ignored outputs tree.
    (ROOT/'outputs/E008_E009_mitosis_audit_details.json').write_text(json.dumps(audit,indent=2)+'\n')
    aggregate=dict(checked_at_utc=audit['checked_at_utc'],events_reviewed=len(events),
        prepared_GT_checksums_verified=True,predictions_changed=False,scope=audit['scope'],
        detailed_annotation_coordinates_kept_local=True,arms={})
    for arm in ('control','E008','E009'):
        counts=dict(events_with_distinct_daughters_within_7um=0,events_with_nearby_final_fork=0,
                    events_recovering_a_previously_missing_daughter=0,recovered_events_with_orphan_daughter=0,
                    previously_missing_daughters_proposed_but_absent_from_final_neighborhood=0)
        for event in events:
            info=event['final'][arm]
            counts['events_with_distinct_daughters_within_7um']+=int(info['distinct_daughter_candidates_within_7um'])
            counts['events_with_nearby_final_fork']+=int(bool(info['nearby_fork_mothers']))
            missing=[i for i,d in enumerate(event['final']['control']['nearest_distances_um'][1:]) if d is None or d>7.]
            recovered=any(info['nearest_distances_um'][i+1] is not None and info['nearest_distances_um'][i+1]<=7. for i in missing)
            counts['events_recovering_a_previously_missing_daughter']+=int(recovered)
            counts['recovered_events_with_orphan_daughter']+=int(recovered and None in info['nearest_daughter_parents'])
            if arm!='control':
                for i in missing:
                    pd=event['proposals'][arm]['nearest_distances_um'][i+1]
                    fd=info['nearest_distances_um'][i+1]
                    counts['previously_missing_daughters_proposed_but_absent_from_final_neighborhood']+=int(pd is not None and pd<=7. and (fd is None or fd>7.))
        aggregate['arms'][arm]=counts
    (ROOT/'results/E008_E009_mitosis_audit.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    for event in events:
        print(event['dataset'],event['mother_frame'],{arm:(info['nearest_distances_um'],info['nearest_daughter_parents']) for arm,info in event['final'].items()})


if __name__=='__main__':main()
