"""E010: reconstruct centers and division paths, then score the final graph."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from notebook_runner import competition_dir, diagnostic_data
from biohub_lab.joint_lineage import JointConfig, ImageEvidence, reconstruct
from biohub_lab.submission import COLUMNS, read_and_validate
from biohub_lab.evaluate import shapes_for, evaluate_csv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_root(slug, required, input_root=Path('/kaggle/input')):
    roots=[input_root/slug,input_root/'notebooks/jarturo'/slug]
    found=[p for p in roots if (p/required).is_file()]
    if len(found)!=1:raise ValueError(f'Expected one completed input {slug}: {found}')
    return found[0]


def verify_inputs(manifest, input_root=Path('/kaggle/input')):
    """Verify exact completed artifacts before any reconstruction work."""
    control_spec=manifest['control']
    control_root=source_root(control_spec['slug'],'run_receipt.json',input_root)
    if sha256(control_root/'run_receipt.json')!=control_spec['receipt_sha256']:
        raise ValueError('Unexpected control receipt version')
    control=json.loads((control_root/'run_receipt.json').read_text())
    control_csv=control_root/'biohub_control/submission.csv'
    if sha256(control_csv)!=control_spec['csv_sha256'] or control_spec['csv_sha256']!=control['arms']['control']['sha256']:
        raise ValueError('Control CSV digest mismatch')
    names=set(manifest['datasets'])
    if names!=set(control['datasets']):raise ValueError('Control datasets differ')
    roots=[]
    for spec in manifest['proposals']:
        source=source_root(spec['slug'],'detector_experiment/result.json',input_root)
        result_file=source/'detector_experiment/result.json'
        if sha256(result_file)!=spec['receipt_sha256']:raise ValueError('Unexpected proposal receipt version')
        result=json.loads(result_file.read_text())
        if result['status']!='complete' or result['method']!=spec['method'] or set(result['datasets'])!=names:
            raise ValueError('Incomplete or incompatible proposal input')
        if set(spec['files'])!=names:raise ValueError('Missing pinned proposal file')
        proposal_root=source/'detector_experiment/proposals'
        for name,expected in spec['files'].items():
            if sha256(proposal_root/(name+'.npz'))!=expected['sha256']:
                raise ValueError(f'Proposal checksum mismatch: {spec["method"]}/{name}')
            original=result['proposal_receipts'][name]
            if original['sha256']!=expected['sha256'] or original['proposals']!=expected['count']:
                raise ValueError('Proposal receipt and pin disagree')
        roots.append(proposal_root)
    return control_csv,control,roots


def graph_arrays(nodes, edges):
    ordered=sorted(nodes);lookup={node:i for i,node in enumerate(ordered)}
    coords=np.array([[nodes[node][key] for key in ('t','z','y','x')] for node in ordered],np.int64)
    linked=np.array([(lookup[s],lookup[t]) for s,t in edges],np.int64).reshape(-1,2)
    return coords,linked


def load_proposals(roots,name,shape,manifest):
    sources=[]
    for root,spec in zip(roots,manifest['proposals']):
        with np.load(root/(name+'.npz'),allow_pickle=False) as values:
            if tuple(values['shape'])!=tuple(shape):raise ValueError('Proposal volume shape mismatch')
            points=values['coords'];scores=values['scores']
        if len(points)!=spec['files'][name]['count']:raise ValueError('Proposal count mismatch')
        sources.append((points,scores))
    return sources


def write_csv(path, graphs):
    index=0
    with Path(path).open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(COLUMNS)
        for name,(coords,edges) in sorted(graphs.items()):
            for node,(t,z,y,x) in enumerate(coords):
                writer.writerow([index,name,'node',node,int(t),int(z),int(y),int(x),-1,-1]);index+=1
            for source,target in edges:
                writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(source),int(target)]);index+=1


def main(package,root=Path('/kaggle/working/joint_experiment'),input_root=Path('/kaggle/input')):
    package=Path(package);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    receipt=dict(experiment='E010',status='verifying_inputs',config=asdict(JointConfig()),
        scope='Conditional training diagnostic: the four videos are in the public secondary detector training set '
              'and have been used for prior error analysis. Not independent validation or a leaderboard score.',
        annotations_used_for_reconstruction=False,ground_truth_used_only_for_final_evaluation=True,
        neural_training=False,full_harmonic_rerun=False,leaderboard_submitted=False,
        selection='One fixed configuration; no diagnostic threshold sweep or annotated-event restriction.')
    receipt_path=root/'result.json'
    def save():receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        manifest=json.loads((package/'baseline/joint_inputs.json').read_text())
        control_csv,control,proposal_roots=verify_inputs(manifest,input_root)
        receipt['input_manifest_sha256']=sha256(package/'baseline/joint_inputs.json')
        receipt['source_hashes']={str(p.relative_to(package)):sha256(p) for p in
            [package/'src/biohub_lab/joint_lineage.py',package/'scripts/joint_runner.py']}
        data,names=diagnostic_data(competition_dir(),root)
        if set(names)!=set(manifest['datasets']):raise ValueError('Diagnostic selection changed')
        shapes=shapes_for(data);base=read_and_validate(control_csv,shapes)
        receipt.update(status='reconstructing',datasets=names,shapes=shapes,statistics={})
        save();graphs={}
        for name,(nodes,edges) in sorted(base.items()):
            receipt['current_video']=name;save();begin=time.monotonic()
            coords,linked=graph_arrays(nodes,edges)
            proposals=load_proposals(proposal_roots,name,shapes[name],manifest)
            print('JOINT_IMAGE_LOADING',name,flush=True)
            evidence=ImageEvidence.from_zarr(data/(name+'.zarr'))
            image_seconds=time.monotonic()-begin
            selected,new_edges,stats=reconstruct(coords,linked,proposals,shapes[name],evidence,root/name)
            del evidence
            graphs[name]=(selected,new_edges)
            stats.update(seconds=time.monotonic()-begin,image_seconds=image_seconds)
            receipt['statistics'][name]=stats;save()
            print('JOINT_VIDEO_COMPLETE',name,json.dumps(stats),flush=True)
        final=root/'predictions.csv';write_csv(final,graphs)
        parsed=read_and_validate(final,shapes)
        receipt.update(status='evaluating',csv_sha256=sha256(final),validated=True,
            final_counts={name:dict(nodes=len(nodes),edges=len(edges)) for name,(nodes,edges) in parsed.items()})
        save()
        # Annotation graphs are first opened here, after predictions are frozen.
        metrics=evaluate_csv(final,data);metrics['scope']=receipt['scope']
        if sha256(final)!=receipt['csv_sha256']:raise ValueError('Predictions changed during evaluation')
        reference=control['arms']['control']['metrics']['summary']
        receipt.update(status='complete',seconds=time.monotonic()-started,metrics=metrics,control=reference,
            delta={k:metrics['summary'][k]-reference[k] for k in ('score','adj_edge_jaccard','division_jaccard')},
            selected_events=sum(v['selected_events'] for v in receipt['statistics'].values()),
            quality_status='requires_comparison_review')
        receipt.pop('current_video',None);save()
        print('JOINT_RESULT',json.dumps(receipt,indent=2),flush=True)
        return receipt
    except Exception as error:
        receipt.update(failed_stage=receipt['status'],status='failed',seconds=time.monotonic()-started,
                       error_type=type(error).__name__,error=str(error))
        save();(root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main(sys.argv[1])
