"""Verify E010 outputs and audit coverage after evaluation, without changing predictions.

Detailed graph/annotation correspondences stay in ignored outputs. Git receives
aggregate counts only. This reproduces metric aggregation, not graph matching.
"""
import ast
import base64
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
from biohub_lab.joint_lineage import (JointConfig, apply_events, chain, combine_proposals,
    enumerate_windows, graph_index, solve_event_packing)
from biohub_lab.detector_proposals import SCALE
from biohub_lab.submission import read_and_validate
import joint_runner
from verify_detector_results import geometry, official_aggregator

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/e010_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonlines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def forks(edges):return sum(count==2 for count in Counter(int(s) for s,_ in edges).values())


def verify():
    launch=read(ROOT/'results/E010_launch.json');status=read(OUTPUT/'status.json')
    result=read(OUTPUT/'joint_experiment/result.json')
    assert status['status']=='COMPLETE' and result['status']=='complete'
    assert result['config']==asdict(JointConfig())
    notebook,=list((OUTPUT/'source').glob('*.ipynb'))
    code='\n'.join(''.join(c['source']) for c in read(notebook)['cells'] if c['cell_type']=='code')
    assignment=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),name
        for name,digest in result['source_hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        manifest_bytes=archive.read('baseline/joint_inputs.json')
        assert hashlib.sha256(manifest_bytes).hexdigest()==result['input_manifest_sha256']
    manifest=json.loads(manifest_bytes)
    roots={'biohub-lab-official-metric-ab':ROOT/'outputs/diagnostic',
           'biohub-lab-cellect-detector':ROOT/'outputs/e008_v2',
           'biohub-lab-gaussian-deblending':ROOT/'outputs/e009_v2'}
    with patch.object(joint_runner,'source_root',side_effect=lambda slug,required,input_root:roots[slug]):
        control_csv,control,proposal_roots=joint_runner.verify_inputs(manifest)
    final_csv=OUTPUT/'joint_experiment/predictions.csv'
    assert sha(final_csv)==result['csv_sha256']
    base=read_and_validate(control_csv,result['shapes']);final=read_and_validate(final_csv,result['shapes'])
    assert set(base)==set(final)==set(result['datasets'])
    statistics={};graphs={};all_windows={};events_by_video={};artifact_hashes={}
    for name in sorted(base):
        folder=OUTPUT/'joint_experiment'/name
        coords,edges=joint_runner.graph_arrays(*base[name]);graphs[name]=(coords,edges)
        proposals=joint_runner.load_proposals(proposal_roots,name,result['shapes'][name],manifest)
        pool,scores,origin=combine_proposals(coords,proposals,result['shapes'][name])
        with np.load(folder/'candidate_pool.npz',allow_pickle=False) as values:
            np.testing.assert_array_equal(pool,values['coords'])
            np.testing.assert_array_equal(scores,values['confidence'])
            np.testing.assert_array_equal(origin,values['origin'])
        expected=enumerate_windows(coords,edges)
        windows=jsonlines(folder/'window_audit.jsonl');all_windows[name]=windows
        assert [(w['mother'],w['t'],w['orphan']) for w in windows]==[(w['mother'],w['t'],w['orphan']) for w in expected]
        events=jsonlines(folder/'event_audit.jsonl');events_by_video[name]=events
        selected=[i for i,e in enumerate(events) if e['selected']]
        decisions=Counter();used_event_ids=[]
        for window,reference in zip(windows,expected):
            decisions.update(window['reasons'])
            assert sum(window['reasons'].values())==np.prod(window['paths_per_anchor'])
            assert len(window['event_ids'])==min(3,window['reasons'].get('positive',0))
            for i in window['event_ids']:
                event=events[i];used_event_ids.append(i)
                assert event['event_id']==i
                for key,value in reference.items():assert event[key]==value,(name,key)
                assert len(event['paths'])==2 and not set(event['paths'][0])&set(event['paths'][1])
                for branch in (0,1):
                    path=event['paths'][branch]
                    assert len(path)==JointConfig().horizon and path[-1]==reference['anchors'][branch]
                    np.testing.assert_array_equal(pool[path,0],np.arange(window['t']+1,window['t']+5))
                    for node,stage in zip(path[:-1],window['path_stages'][branch]):
                        assert int(pool[node,0])==stage['frame'] and node in stage['ranked_candidates']
                evidence=event['evidence']
                assert evidence['gain']>0 and min(evidence['post_contrast'])>=JointConfig().min_post_contrast
                onset=float(np.median(evidence['post_contrast'])-np.mean(evidence['pre_contrast']))
                np.testing.assert_allclose(onset,evidence['contrast_onset'],atol=1.e-12)
        assert used_event_ids==list(range(len(events)))
        rebuilt,linked,changes=apply_events(coords,edges,pool,events,selected)
        saved=np.load(folder/'final_graph.npz',allow_pickle=False)
        np.testing.assert_array_equal(rebuilt,saved['coords']);np.testing.assert_array_equal(linked,saved['edges'])
        final_coords,final_edges=joint_runner.graph_arrays(*final[name])
        np.testing.assert_array_equal(rebuilt,final_coords);np.testing.assert_array_equal(linked,final_edges)
        optimum,solver=solve_event_packing(events,pool)
        chosen_gain=sum(events[i]['evidence']['gain'] for i in selected)
        np.testing.assert_allclose(chosen_gain,sum(events[i]['evidence']['gain'] for i in optimum),atol=1.e-7)
        receipt=read(folder/'receipt.json');runtime=result['statistics'][name]
        assert all(runtime[k]==v for k,v in receipt.items())
        assert dict(decisions)==receipt['pair_decisions']
        assert changes==receipt['changes']
        assert (len(windows),len(events),len(selected))==(receipt['windows'],receipt['positive_hypotheses'],receipt['selected_events'])
        assert len(rebuilt)==receipt['final_nodes'] and len(linked)==receipt['final_edges']
        assert result['final_counts'][name]==dict(nodes=len(rebuilt),edges=len(linked))
        assert forks(linked)==forks(edges)+len(selected)
        statistics[name]=dict(windows=len(windows),positive_hypotheses=len(events),selected_events=len(selected),
            changes=changes,baseline_forks=forks(edges),final_forks=forks(linked),
            cached_selection_gain=chosen_gain,packing_objective_reproduced=True,
            local_solver_optimal=solver['optimal'])
        artifact_hashes[name]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    summary=official_aggregator()(result['metrics']['samples'])
    for k,v in summary.items():np.testing.assert_allclose(v,result['metrics']['summary'][k],atol=1.e-14)
    for row in result['metrics']['samples']:assert row['num_pred_nodes']==len(final[row['dataset']][0])
    assert result['control']==control['arms']['control']['metrics']['summary']
    for k,v in result['delta'].items():np.testing.assert_allclose(summary[k]-result['control'][k],v,atol=1.e-14)
    totals=dict(nodes=sum(len(n) for n,e in final.values()),edges=sum(len(e) for n,e in final.values()),
        predicted_forks=sum(s['final_forks'] for s in statistics.values()),
        windows=sum(s['windows'] for s in statistics.values()),
        positive_hypotheses=sum(s['positive_hypotheses'] for s in statistics.values()),
        selected_events=sum(s['selected_events'] for s in statistics.values()),
        changes={k:sum(s['changes'][k] for s in statistics.values()) for k in next(iter(statistics.values()))['changes']},
        **{k:sum(s[k] for s in result['metrics']['samples']) for k in ('edge_tp','edge_fp','edge_fn')})
    assert totals['selected_events']==result['selected_events']
    completed=dict(result,version=launch['versionNumber'],kernel=launch['url'],payload_sha256=launch['payload_sha256'],
        checked_at_utc=status['checked_at_utc'],totals=totals,artifact_hashes=artifact_hashes,
        verification=dict(source_payload=True,source_and_manifest_hashes=True,cached_inputs=True,
            candidate_pools=True,window_inventory=True,window_event_membership=True,
            final_graph_reconstructed=True,final_csv=True,packing_objective=True,metric_aggregation=True,
            raw_image_scores_recomputed_locally=False,official_graph_matching_rerun_locally=False),
        quality_status='below_control_same_division_recall_more_false_positives',decision='do_not_submit_this_version')
    (ROOT/'results/E010_completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    return completed,graphs,final,all_windows,events_by_video


def coverage(completed,graphs,final,windows,events):
    """Exact-frame post-hoc coverage; not official matching or an oracle score."""
    details=[];causes=Counter();nearby_covered=0
    truth_events=[]
    for name in graphs:
        path=ROOT/'outputs/mitosis_audit/temporal_cache'/name/'graph.npz'
        receipt=read(path.parent/'receipt.json')
        with np.load(path,allow_pickle=False) as values:gt={key:values[key] for key in values.files}
        digest=hashlib.sha256(json.dumps(receipt['config'],sort_keys=True).encode()+
            gt['coords'][:receipt['gt_nodes']].tobytes()+gt['edges'].tobytes()).hexdigest()
        assert digest==receipt['config_hash']
        for i in np.flatnonzero(gt['triple_labels']==1):truth_events.append((name,gt['coords'][gt['triples'][i]]))
    assert len(truth_events)==7
    for name,truth in truth_events:
        coords,edges=graphs[name];new,links=joint_runner.graph_arrays(*final[name])
        before=geometry(coords,truth,edges);after=geometry(new,truth,links)
        mother=before['nearest_indices'][0];children,parents=graph_index(coords,edges)
        exact=[w for w in windows[name] if w['mother']==mother]
        near=[w for w in windows[name] if w['t']==int(truth[0,0]) and
              np.linalg.norm((coords[w['mother'],1:]-truth[0,1:])*SCALE)<=7.]
        reason='window_present'
        if before['nearby_fork_mothers']:reason='already_has_nearby_fork'
        elif mother not in parents or parents[mother] not in parents:reason='missing_two_parent_context'
        elif chain(mother,6,children) is None:reason='primary_chain_not_single_child_for_six_steps'
        elif not exact:reason='no_eligible_persistent_orphan_anchor'
        if not before['nearby_fork_mothers']:
            causes[reason]+=1;nearby_covered+=int(bool(near))
        details.append(dict(dataset=name,gt_coords=truth.tolist(),baseline=before,final=after,
            nearest_mother_windows=len(exact),all_mothers_within_7um_exact_frame_windows=len(near),
            candidate_coverage_reason=reason))
    scoped='Post-hoc exact-frame geometry at 7um for seven annotated divisions; not official graph matching. '
    scoped+='No thresholds or predictions changed. Other times are not counted by this coverage diagnostic.'
    (ROOT/'outputs/E010_coverage_details.json').write_text(json.dumps(dict(scope=scoped,events=details),indent=2)+'\n')
    audit=dict(experiment='E010',checked_at_utc=datetime.now(timezone.utc).isoformat(),scope=scoped,
        annotated_events_reviewed=len(details),prepared_gt_checksums_verified=True,
        baseline_events_with_nearby_fork=sum(bool(e['baseline']['nearby_fork_mothers']) for e in details),
        final_events_with_nearby_fork=sum(bool(e['final']['nearby_fork_mothers']) for e in details),
        baseline_events_without_nearby_fork=sum(causes.values()),
        missing_events_with_any_candidate_mother_within_7um_exact_frame=nearby_covered,
        nearest_mother_rejection_causes=dict(causes),
        baseline_distinct_daughter_events=sum(e['baseline']['distinct_daughter_candidates_within_7um'] for e in details),
        final_distinct_daughter_events=sum(e['final']['distinct_daughter_candidates_within_7um'] for e in details),
        predictions_changed=False,ground_truth_used_for_inference=False,detailed_coordinates_kept_local=True)
    (ROOT/'results/E010_coverage_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


if __name__=='__main__':
    completed,graphs,final,windows,events=verify()
    audit=coverage(completed,graphs,final,windows,events)
    print(json.dumps(dict(score=completed['metrics']['summary'],delta=completed['delta'],totals=completed['totals'],coverage=audit),indent=2))
