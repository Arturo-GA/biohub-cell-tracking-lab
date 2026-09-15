"""Verify E011 graphs and reproduce development coverage without changing them.

Detailed annotation correspondences remain under ignored outputs/. Reports in
results/ contain aggregate counts only. No proposal, threshold or score is tuned.
"""
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

from biohub_lab.detection_dag import (DAGConfig,build_dag,merge_sources,pair_mask,
    sample_trajectories,select_development)
from biohub_lab.dag_coverage import CoverageGate,evaluate_graph,readiness
from biohub_lab.detector_proposals import SCALE
from biohub_lab.cellect_detector import WEIGHT_SHA,MODEL_SHA,CELLECT_CONFIG
from biohub_lab.gaussian_detector import GAUSSIAN_CONFIG

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/e011_v1'
RUN=OUTPUT/'detection_dag_experiment'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as values:return {key:values[key] for key in values.files}


def verify_source(result):
    status=read(OUTPUT/'status.json');launch=read(ROOT/'results/E011_launch.json')
    assert status['status']=='COMPLETE' and result['status']=='complete'
    notebook,=list((OUTPUT/'source').glob('*.ipynb'))
    code='\n'.join(''.join(c['source']) for c in read(notebook)['cells'] if c['cell_type']=='code')
    assignment=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    payload=base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    assert hashlib.sha256(payload).hexdigest()==launch['payload_sha256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.endswith('.py') or name=='baseline/detection_dag_dev.json':
                assert archive.read(name)==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),name
        for name,digest in result['source_hashes'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        manifest_bytes=archive.read('baseline/detection_dag_dev.json')
        assert hashlib.sha256(manifest_bytes).hexdigest()==result['development_manifest_sha256']
        assert hashlib.sha256(archive.read('src/biohub_cellect/model.py')).hexdigest()==MODEL_SHA
        assert len(archive.read('licenses/CELLECT.txt'))>10000
    manifest=json.loads(manifest_bytes)
    cache_path=ROOT/'outputs/temporal_prepare/temporal_cache/cache_manifest.json'
    assert sha(cache_path)==read(ROOT/'results/E004_prepared.json')['manifest_sha256']
    cache=read(cache_path);assert cache['complete']
    names=sorted(row['name'] for row in cache['videos'])
    assert hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()==manifest['training_names_sha256']
    assert select_development(names,manifest['excluded'],manifest['per_group'])==manifest['development']==result['datasets']
    assert not set(result['datasets'])&set(manifest['excluded']) and len(result['datasets'])==48
    assert result['config']==asdict(DAGConfig())==manifest['config']
    assert result['gate']==asdict(CoverageGate())==manifest['gate']
    meta=read(OUTPUT/'source/kernel-metadata.json')
    assert not meta['kernel_sources'] and meta['is_private']
    return launch,status,{row['name']:row for row in cache['videos']}


def validate_neighbor_table(coords,actual,expected,radius):
    """Accept only equal-distance boundary ties across SciPy builds."""
    assert actual.shape==expected.shape
    assert np.all(actual>=-1) and np.all(actual<len(coords))
    valid=actual>=0
    assert np.array_equal(valid.sum(1),(expected>=0).sum(1))
    assert np.all(coords[actual[valid],0]==np.repeat(coords[:,0],valid.sum(1))+1)
    # Padding follows all real neighbors; each row references distinct cells.
    assert np.all(np.diff(valid.astype(int),axis=1)<=0)
    sorted_ids=np.sort(np.where(valid,actual,len(coords)+np.arange(actual.shape[1])),axis=1)
    assert not np.any(np.diff(sorted_ids,axis=1)==0)
    def distances(table):
        dist=np.linalg.norm((coords[table.clip(0),1:]-coords[:,None,1:])*SCALE,axis=-1)
        return np.sort(np.where(table>=0,dist,np.inf),axis=1)
    ds=distances(actual);de=distances(expected)
    assert np.all(ds[np.isfinite(ds)]<=radius+1.e-10)
    np.testing.assert_allclose(ds,de,atol=1.e-10,rtol=0)


def detection_geometry(coords,points):
    near=[];distances=[]
    for point in points:
        ids=np.flatnonzero(coords[:,0]==point[0]);d=np.linalg.norm((coords[ids,1:]-point[1:])*SCALE,axis=1)
        near.append(set(map(int,ids[d<=CoverageGate().match_um])))
        distances.append(float(d.min()) if len(d) else None)
    return dict(mother=bool(near[0]),distinct_daughters=any(a!=b for a in near[1] for b in near[2]),
                nearest_um=distances)


def main():
    result=read(RUN/'result.json');launch,status,cache_records=verify_source(result)
    frozen=read(RUN/'frozen_inputs.json')
    assert sha(RUN/'frozen_inputs.json')==result['frozen_inputs_sha256']
    assert frozen['all_detection_graphs_complete'] and not frozen['annotations_read']
    assert frozen['datasets']==result['datasets']
    workers=[read(p) for p in sorted(RUN.glob('worker_*.json'))]
    completed=[name for worker in workers for name in worker['completed']]
    assert sorted(completed)==result['datasets'] and len(completed)==len(set(completed))
    assert all(w['status']=='complete' and not w['annotations_read'] for w in workers)
    all_counts={};exact=0;tie_equivalent=0;total_sources=Counter();bottlenecks=Counter()
    source_coverage=Counter();details=[];reasons=Counter();full_failure=Counter();gt_hashes={}
    for index,name in enumerate(result['datasets'],1):
        record=frozen['records'][name];folder=RUN/'videos'/name
        for relative,digest in record['files'].items():assert sha(RUN/relative)==digest,(name,relative)
        generation=read(folder/'generation.json');assert generation==record['generation']
        small=read(folder/'dag/receipt.json');assert all(generation[k]==v for k,v in small.items())
        assert not generation['annotations_read'] and not generation['reference_graph_used']
        sources=[];source_coords={};shape=None
        for method in ('cellect','gaussian'):
            path=folder/(method+'.npz');data=load(path);receipt=read(path.with_suffix('.json'))
            assert sha(path)==receipt['sha256']==generation['proposal_files'][method]['sha256']
            assert len(data['coords'])==receipt['proposals']==generation['proposal_files'][method]['count']
            assert [int((data['coords'][:,0]==t).sum()) for t in range(data['shape'][0])]==receipt['frame_counts']
            if method=='cellect':
                assert receipt['weight_sha256']==WEIGHT_SHA and receipt['architecture_sha256']==MODEL_SHA
                assert receipt['detector_config']==CELLECT_CONFIG
            else:assert receipt['detector_config']==GAUSSIAN_CONFIG
            if shape is not None:assert np.array_equal(shape,data['shape'])
            shape=data['shape'];sources.append((data['coords'],data['scores']))
            source_coords[method]=np.unique(np.rint(data['coords']).astype(np.int32),axis=0)
            total_sources[method]+=len(data['coords'])
        coords,scores,origin=merge_sources(sources,shape)
        expected=build_dag(coords,scores,origin,shape);graph=load(folder/'dag/graph.npz')
        assert set(expected)==set(graph)
        for key in ('coords','scores','origin','shape','pair_slots'):np.testing.assert_array_equal(graph[key],expected[key])
        identical=all(np.array_equal(graph[key],expected[key]) for key in graph)
        exact+=int(identical);tie_equivalent+=int(not identical)
        for key,radius in [('daughters',DAGConfig().mother_radius_um),('continuation',DAGConfig().continuation_um)]:
            validate_neighbor_table(coords,graph[key],expected[key],radius)
        slots,bits,counts=pair_mask(coords,graph['daughters'])
        np.testing.assert_array_equal(bits,graph['pair_bits']);np.testing.assert_array_equal(counts,graph['pair_counts'])
        triple_sample,path_sample=sample_trajectories(graph,name)
        samples=load(folder/'dag/trajectory_samples.npz')
        np.testing.assert_array_equal(samples['triples'],triple_sample);np.testing.assert_array_equal(samples['paths'],path_sample)
        metrics=dict(nodes=len(coords),mother_daughter_edges=int((graph['daughters']>=0).sum()),
            continuation_edges=int((graph['continuation']>=0).sum()),division_pairs=int(counts.sum()),
            mothers_with_pairs=int((counts>0).sum()),sample_queries=len(triple_sample),
            sample_queries_with_paths=int((path_sample[:,0,0]>=0).sum()) if len(path_sample) else 0)
        for key,value in metrics.items():assert value==generation[key]
        gt_root=ROOT/'outputs/e011_gt_audit/temporal_cache'/name
        gt=load(gt_root/'graph.npz');gt_receipt=read(gt_root/'receipt.json')
        assert gt_receipt==cache_records[name]
        truth=gt['coords'][:gt_receipt['gt_nodes']];edges=gt['edges']
        assert edges.size==0 or edges.max()<len(truth)
        digest=hashlib.sha256(json.dumps(gt_receipt['config'],sort_keys=True).encode()+truth.tobytes()+edges.tobytes()).hexdigest()
        assert digest==gt_receipt['config_hash'];gt_hashes[name]=sha(gt_root/'graph.npz')
        local,rows=evaluate_graph(graph,truth,edges)
        original=read(RUN/'coverage'/(name+'.json'))
        assert local==original['counts']==result['readiness']['per_video'][name]
        assert rows==original['annotation_conditioned_witnesses']
        assert local['events']==gt_receipt['division_positive']
        all_counts[name]=local
        children=defaultdict(list)
        for s,t in edges:children[int(s)].append(int(t))
        source_coords['union_before_merge']=np.unique(np.rint(np.concatenate([v[0] for v in sources])).astype(np.int32),axis=0)
        source_coords['merged']=coords
        for row in rows:
            mother=row['annotated_mother_index'];kids=children[mother];points=truth[[mother,*kids]]
            geometry={method:detection_geometry(value,points) for method,value in source_coords.items()}
            for method,info in geometry.items():source_coverage[method]+=int(info['mother'] and info['distinct_daughters'])
            info=geometry['merged'];reasons[row['reason']]+=1
            if not row['centers']:
                reason='both_mother_and_daughters_missing' if not info['mother'] and not info['distinct_daughters'] else 'mother_missing' if not info['mother'] else 'daughters_missing_or_not_distinct'
                bottlenecks[reason]+=1
            full_reason=None
            if row['full_annotation_context'] and row['annotated_path_witness'] is None:
                a,b=kids;support=bool(info['mother'])
                for step in range(4):
                    pair=detection_geometry(coords,truth[[mother,a,b]])
                    support=bool(support and pair['distinct_daughters'])
                    if step<3:a=children[a][0];b=children[b][0]
                full_reason='hypothesis_constraints_despite_centers_present' if support else 'missing_centers_in_annotated_temporal_context'
                full_failure[full_reason]+=1
            details.append(dict(dataset=name,gt_coords=points.tolist(),coverage=row,source_geometry=geometry,
                failed_annotated_path_cause=full_reason))
        print('E011_VERIFIED',index,'/ 48',name,'divisions',local['events'],flush=True)
    ready=readiness(all_counts);ready['per_video']=all_counts
    assert ready==result['readiness']==read(RUN/'readiness.json')
    totals={key:sum(frozen['records'][name]['generation'][key] for name in result['datasets']) for key in result['totals']}
    assert totals==result['totals']
    checks=dict(source_payload=True,source_and_manifest_hashes=True,development_selection=True,
        all_frozen_file_hashes=True,detector_receipts=True,merged_detection_arrays=True,
        graph_reconstruction_exact_videos=exact,graph_equivalent_neighbor_ties_videos=tie_equivalent,
        pair_masks=True,sampled_paths=True,ground_truth_cache_checksums=True,coverage_counts_recomputed=True,
        all_annotation_witnesses_reproduced=True,readiness_gate_recomputed=True,
        raw_image_detector_inference_rerun_locally=False)
    completed=dict(result,version=launch['versionNumber'],kernel=launch['url'],payload_sha256=launch['payload_sha256'],
        checked_at_utc=status['checked_at_utc'],verification=checks,proposal_totals=dict(total_sources),
        quality_status='coverage_gate_failed_detection_support_insufficient',decision='do_not_train_or_submit_this_candidate_pool')
    (ROOT/'results/E011_completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    scope='Post-hoc audit at the fixed 7um match radius on the 48 development videos. Coverage ceilings only, not accuracy or official metric. No predictions or thresholds changed.'
    audit=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),scope=scope,events_reviewed=len(details),
        failure_stage_counts=dict(reasons),missing_center_breakdown=dict(bottlenecks),
        center_coverage_by_source=dict(source_coverage),failed_full_context_paths=dict(full_failure),
        ground_truth_coordinates_kept_local=True,predictions_changed=False)
    (ROOT/'results/E011_bottleneck_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (ROOT/'outputs/E011_bottleneck_details.json').write_text(json.dumps(dict(scope=scope,events=details,gt_graph_sha256=gt_hashes),indent=2)+'\n')
    print(json.dumps(dict(readiness={k:v for k,v in ready.items() if k!='per_video'},audit=audit,verification=checks),indent=2))


if __name__=='__main__':main()
