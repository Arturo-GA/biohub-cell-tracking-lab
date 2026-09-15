"""Reconstruct E012 graphs, reproduce paired coverage and report aggregates only."""
import ast
import base64
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from biohub_lab.detection_dag import DAGConfig,build_dag,pair_mask,sample_trajectories,select_development
from biohub_lab.dag_coverage import CoverageGate,evaluate_graph,readiness
from biohub_lab.harmonic_centers import CHECKPOINTS,DETECTOR_CONFIG,detector_source,merge_primary
from verify_detection_dag_result import validate_neighbor_table,detection_geometry

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/e012_v1';RUN=OUT/'harmonic_dag_experiment'
CACHED=ROOT/'outputs/e011_v1/detection_dag_experiment'
ARMS=('e011','harmonic','combined')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def verify_source(result):
    launch=read(ROOT/'results/E012_launch.json');status=read(OUT/'status.json')
    assert status['status']=='COMPLETE' and result['status']=='complete'
    notebook,=list((OUT/'source').glob('*.ipynb'))
    code='\n'.join(''.join(c['source']) for c in read(notebook)['cells'] if c['cell_type']=='code')
    node=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),name
        for name,digest in result['source_hashes'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        manifest=archive.read('baseline/detection_dag_dev.json');pins=archive.read('baseline/harmonic_dag_inputs.json')
        assert hashlib.sha256(manifest).hexdigest()==result['development_manifest_sha256']
        assert hashlib.sha256(pins).hexdigest()==result['cached_input_manifest_sha256']
    manifest=json.loads(manifest);pins=json.loads(pins)
    assert manifest['config']==result['config']==asdict(DAGConfig())
    assert manifest['gate']==result['gate']==asdict(CoverageGate())
    assert result['detector_config']==DETECTOR_CONFIG
    path=ROOT/'outputs/temporal_prepare/temporal_cache/cache_manifest.json'
    assert sha(path)==read(ROOT/'results/E004_prepared.json')['manifest_sha256']
    cache=read(path);assert cache['complete'];names=sorted(r['name'] for r in cache['videos'])
    assert hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()==manifest['training_names_sha256']
    assert select_development(names,manifest['excluded'],manifest['per_group'])==manifest['development']==pins['datasets']==result['datasets']
    assert len(result['datasets'])==48 and not set(result['datasets'])&set(manifest['excluded'])
    meta=read(OUT/'source/kernel-metadata.json')
    assert meta['is_private'] and meta['kernel_sources']==['jarturo/biohub-lab-detection-dag-coverage']
    runtime=read(RUN/'runtime.json');assert result['runtime']==runtime
    assert sha(RUN/'image_detector.py')==runtime['detector_source_sha256']
    assets=ROOT/'artifacts/e012_detector_check'
    source=detector_source((assets/'smoke/repo/scripts/predict_unet_transformer.py').read_text(),
        (assets/'repo/scripts/train_unet_transformer.py').read_text())
    assert source==(RUN/'image_detector.py').read_text()
    patched=(assets/'smoke/repo/scripts/predict_unet_transformer.py').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(patched).hexdigest()==runtime['patched_predictor_sha256']
    assert sha(assets/'repo/scripts/train_unet_transformer.py')==runtime['trainer_sha256']
    assert runtime['checkpoint_sha256']==CHECKPOINTS
    integrity=read(OUT/'bidirectional_production_runtime_integrity.json')
    assert not integrity['ground_truth_accessed'] and integrity['verified_before_dynamic_source_patch']
    baseline=ast.parse((ROOT/'baseline/harmonic_inference.py').read_text())
    expected=next(ast.literal_eval(n.value) for n in baseline.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_support_expected_sha256' for t in n.targets))
    assert integrity['support_repo_python_sha256']==expected
    for key,digest in CHECKPOINTS.items():
        assert sha(assets/key/'edge_predictor_best.pth')==digest==integrity['checkpoint_sha256'][key]
        assert sha(assets/key/'config.json')==runtime['config_sha256'][key]
    assert sha(CACHED/'frozen_inputs.json')==pins['frozen_sha256']
    for name,files in pins['files'].items():
        for relative,digest in files.items():assert sha(CACHED/'videos'/name/relative)==digest,(name,relative)
    return launch,status,pins,{r['name']:r for r in cache['videos']}


def verify_graph(graph,expected,folder,name,receipt):
    assert set(graph)==set(expected)
    for key in ('coords','scores','origin','shape','pair_slots'):np.testing.assert_array_equal(graph[key],expected[key])
    identical=all(np.array_equal(graph[k],expected[k]) for k in graph)
    for key,radius in [('daughters',DAGConfig().mother_radius_um),('continuation',DAGConfig().continuation_um)]:
        validate_neighbor_table(graph['coords'],graph[key],expected[key],radius)
    slots,bits,counts=pair_mask(graph['coords'],graph['daughters'])
    np.testing.assert_array_equal(bits,graph['pair_bits']);np.testing.assert_array_equal(counts,graph['pair_counts'])
    triples,paths=sample_trajectories(graph,name);samples=load(folder/'trajectory_samples.npz')
    np.testing.assert_array_equal(triples,samples['triples']);np.testing.assert_array_equal(paths,samples['paths'])
    metrics=dict(nodes=len(graph['coords']),mother_daughter_edges=int((graph['daughters']>=0).sum()),
        continuation_edges=int((graph['continuation']>=0).sum()),division_pairs=int(counts.sum()),
        mothers_with_pairs=int((counts>0).sum()),sample_queries=len(triples),
        sample_queries_with_paths=int((paths[:,0,0]>=0).sum()) if len(paths) else 0)
    for key,value in metrics.items():assert receipt[key]==value
    assert receipt==read(folder/'receipt.json') and receipt['graph_sha256']==sha(folder/'graph.npz')
    return identical


def main():
    result=read(RUN/'result.json');launch,status,pins,cache_records=verify_source(result)
    frozen=read(RUN/'frozen_inputs.json')
    assert sha(RUN/'frozen_inputs.json')==result['frozen_inputs_sha256']
    assert not frozen['annotations_read'] and frozen['all_detection_graphs_complete']
    assert frozen['datasets']==result['datasets']
    workers=[read(p) for p in sorted(RUN.glob('worker_*.json'))]
    completed=[name for worker in workers for name in worker['completed']]
    assert sorted(completed)==result['datasets'] and len(completed)==len(set(completed))
    assert all(w['status']=='complete' and not w['annotations_read'] for w in workers)
    statistics={arm:{} for arm in ARMS};exact=Counter();ties=Counter();primary_count=0;added_count=0
    transitions={base:{stage:Counter() for stage in ('centers','pairs','paths','annotated_paths')} for base in ('e011','harmonic')}
    failure=Counter();primary_failure=Counter();details=[];gt_hashes={}
    for index,name in enumerate(result['datasets'],1):
        folder=RUN/'videos'/name;record=frozen['records'][name];generation=read(folder/'generation.json')
        assert record['generation']==generation and not generation['annotations_read'] and not generation['reference_graph_used']
        for relative,digest in record['files'].items():assert sha(RUN/relative)==digest,(name,relative)
        primary=load(folder/'harmonic.npz');receipt=read(folder/'harmonic.json')
        assert receipt==generation['primary'] and receipt['sha256']==sha(folder/'harmonic.npz')
        assert not receipt['annotations_read'] and not receipt['reference_graph_used']
        assert receipt['checkpoint_sha256']==CHECKPOINTS and receipt['detector_source_sha256']==result['runtime']['detector_source_sha256']
        assert receipt['detector_config']==DETECTOR_CONFIG and receipt['window_size']==2 and receipt['downsample']==[1,4,4]
        coords=primary['coords'];shape=primary['shape'];assert shape.tolist()==generation['shape']
        assert len(coords)==receipt['proposals']==generation['primary_centers_preserved']
        assert [int((coords[:,0]==t).sum()) for t in range(shape[0])]==receipt['frame_counts']
        np.testing.assert_array_equal(primary['scores'],np.ones(len(coords)))
        auxiliary=[]
        for method in ('cellect','gaussian'):
            path=CACHED/'videos'/name/(method+'.npz');values=load(path)
            assert sha(path)==generation['cached_proposal_sha256'][method]==pins['files'][name][method+'.npz']
            np.testing.assert_array_equal(values['shape'],shape);auxiliary.append((values['coords'],values['scores']))
        combined=merge_primary(coords,auxiliary,shape)
        np.testing.assert_array_equal(combined[0][:len(coords)],coords)
        assert generation['added_auxiliary_centers']==len(combined[0])-len(coords)
        primary_count+=len(coords);added_count+=generation['added_auxiliary_centers']
        graphs={'e011':load(CACHED/'videos'/name/'dag/graph.npz')}
        assert record['baseline_graph_sha256']==pins['files'][name]['dag/graph.npz']
        for arm,values in [('harmonic',(coords,np.ones(len(coords)),np.zeros(len(coords),np.int8))),('combined',combined)]:
            graph=load(folder/arm/'graph.npz');expected=build_dag(*values,shape)
            identical=verify_graph(graph,expected,folder/arm,name,generation['arms'][arm])
            exact[arm]+=int(identical);ties[arm]+=int(not identical);graphs[arm]=graph
        gt_root=ROOT/'outputs/e011_gt_audit/temporal_cache'/name
        gt=load(gt_root/'graph.npz');gt_receipt=read(gt_root/'receipt.json')
        assert gt_receipt==cache_records[name]
        truth=gt['coords'][:gt_receipt['gt_nodes']];edges=gt['edges']
        assert edges.size==0 or edges.max()<len(truth)
        assert hashlib.sha256(json.dumps(gt_receipt['config'],sort_keys=True).encode()+truth.tobytes()+edges.tobytes()).hexdigest()==gt_receipt['config_hash']
        gt_hashes[name]=sha(gt_root/'graph.npz');rows={}
        for arm in ARMS:
            counts,witnesses=evaluate_graph(graphs[arm],truth,edges)
            original=read(RUN/'coverage'/arm/(name+'.json'))
            assert counts==original['counts']==result['arms'][arm]['per_video'][name]
            assert witnesses==original['annotation_conditioned_witnesses']
            assert counts['events']==gt_receipt['division_positive']
            statistics[arm][name]=counts;rows[arm]={r['annotated_mother_index']:r for r in witnesses}
        children=defaultdict(list)
        for s,t in edges:children[int(s)].append(int(t))
        for mother,row in rows['combined'].items():
            def flag(r,stage):
                return bool(r['centers']) if stage=='centers' else bool(r['candidate_triples']) if stage=='pairs' else r['path_witness'] is not None if stage=='paths' else r['annotated_path_witness'] is not None
            for base in transitions:
                for stage in transitions[base]:
                    if stage=='annotated_paths' and not row['full_annotation_context']:continue
                    before=flag(rows[base][mother],stage);after=flag(row,stage)
                    transitions[base][stage]['retained' if before and after else 'gained' if after else 'lost' if before else 'still_missing']+=1
            points=truth[[mother,*children[mother]]]
            info={arm:detection_geometry(graphs[arm]['coords'],points) for arm in ARMS}
            if not rows['harmonic'][mother]['centers']:
                x=info['harmonic'];primary_failure['both_missing' if not x['mother'] and not x['distinct_daughters'] else 'mother_missing' if not x['mother'] else 'daughters_missing_or_not_distinct']+=1
            cause=None
            if row['full_annotation_context'] and row['annotated_path_witness'] is None:
                a,b=children[mother];support=info['combined']['mother']
                for step in range(DAGConfig().horizon):
                    support=bool(support and detection_geometry(graphs['combined']['coords'],truth[[mother,a,b]])['distinct_daughters'])
                    if step<DAGConfig().horizon-1:a=children[a][0];b=children[b][0]
                cause='hypothesis_constraints_despite_centers_present' if support else 'missing_centers_in_annotated_temporal_context';failure[cause]+=1
            details.append(dict(dataset=name,annotated_coords=points.tolist(),witnesses={arm:rows[arm][mother] for arm in ARMS},
                initial_detection_geometry=info,combined_remaining_failure_cause=cause))
        print('E012_VERIFIED',index,'/ 48',name,'divisions',statistics['combined'][name]['events'],flush=True)
    evaluated={arm:dict(readiness(statistics[arm]),per_video=statistics[arm]) for arm in ARMS}
    assert evaluated==result['arms']==read(RUN/'readiness.json')
    assert evaluated['e011']==read(ROOT/'results/E011_completed.json')['readiness']
    assert {k:evaluated['combined']['totals'][k]-evaluated['e011']['totals'][k] for k in evaluated['e011']['totals']}==result['delta_vs_e011']
    for arm in ('harmonic','combined'):
        totals={k:sum(r['generation']['arms'][arm][k] for r in frozen['records'].values()) for k in result['totals'][arm]}
        assert totals==result['totals'][arm]
    checks=dict(source_payload=True,source_and_manifest_hashes=True,actual_detector_source_reproduced=True,
        checkpoint_and_config_hashes=True,runtime_integrity_receipt=True,development_selection=True,
        cached_E011_files=240,new_frozen_files=sum(len(r['files']) for r in frozen['records'].values()),
        graph_reconstruction_exact_videos=dict(exact),graph_equivalent_neighbor_ties_videos=dict(ties),
        all_primary_centers_preserved=True,pair_masks=True,sampled_paths=True,ground_truth_cache_checksums=True,
        coverage_counts_and_witnesses_recomputed_all_arms=True,E011_previous_result_reproduced=True,
        coverage_gates_recomputed=True,full_image_detector_inference_rerun_locally=False)
    completed=dict(result,version=launch['versionNumber'],kernel=launch['url'],payload_sha256=launch['payload_sha256'],
        checked_at_utc=status['checked_at_utc'],verification=checks,primary_centers_preserved=primary_count,
        added_auxiliary_centers=added_count,quality_status='combined_coverage_gate_passed_accuracy_not_yet_measured',
        decision='proceed_to_event_scoring_research_not_ready_for_submission')
    (ROOT/'results/E012_completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    audit=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),events_reviewed=len(details),
        scope='Paired post-hoc audit at the fixed 7um radius on 48 development videos. Optimistic coverage, not accuracy; no prediction or threshold changed.',
        paired_coverage_transitions={base:{stage:dict(counts) for stage,counts in stages.items()} for base,stages in transitions.items()},
        harmonic_initial_missing_centers=dict(primary_failure),combined_remaining_full_context_failures=dict(failure),
        ground_truth_correspondences_kept_local=True,predictions_changed=False)
    (ROOT/'results/E012_coverage_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (ROOT/'outputs/E012_coverage_details.json').write_text(json.dumps(dict(events=details,gt_graph_sha256=gt_hashes),indent=2)+'\n')
    print(json.dumps(dict(arms={arm:{k:v for k,v in item.items() if k!='per_video'} for arm,item in evaluated.items()},verification=checks,audit=audit),indent=2))


if __name__=='__main__':main()
