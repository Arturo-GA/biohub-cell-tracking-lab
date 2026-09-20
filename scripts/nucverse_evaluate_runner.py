"""E024 CPU instance grouping and honest sparse annotation coverage diagnostics."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from association_cpu_runner import locate
from biohub_lab.nucverse_instances import centers_from_fields,matched_truth

def main(package):
    root=Path('/kaggle/working/nucverse_evaluation');root.mkdir(exist_ok=True);start=time.monotonic()
    inputs=Path('/kaggle/input');fields_root=locate(inputs,'nucverse_fields/result.json').parent
    field_result=json.loads((fields_root/'result.json').read_text());assert field_result['status']=='complete'
    references=locate(inputs,'nucverse_inputs/result.json').parent;reports=[];frozen=[]
    # Decode all predictions before reading any annotation references.
    for frame in field_result['frames']:
        path=fields_root/frame['output'];assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['output_sha256']
        fields=np.load(path,allow_pickle=False);centers,diagnostic=centers_from_fields(fields)
        target=root/(Path(frame['file']).stem+'_centers.npy');np.save(target,centers,allow_pickle=False)
        frozen.append(dict(frame=frame,prediction=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),decoding=diagnostic))
        print('DECODED',frame['file'],len(centers),flush=True)
    (root/'frozen_predictions.json').write_text(json.dumps(dict(annotations_read=False,frames=frozen),indent=2)+'\n')
    for item in frozen:
        frame=item['frame'];prediction=np.load(root/item['prediction'],allow_pickle=False)
        with np.load(references/(Path(frame['file']).stem+'_reference.npz'),allow_pickle=False) as data:truth=data['truth'];harmonic=data['harmonic']
        old=matched_truth(harmonic,truth);new=matched_truth(prediction,truth);union=matched_truth(np.concatenate([harmonic,prediction]),truth)
        reports.append(dict(video=frame['video'],frame=frame['frame'],gt_nodes=len(truth),harmonic_nodes=len(harmonic),nucverse_nodes=len(prediction),
            harmonic_matched=len(old),nucverse_matched=len(new),nucverse_only=len(new-old),harmonic_only=len(old-new),
            union_coverage_upper_bound=len(union),decoding=item['decoding']))
    totals={k:sum(r[k] for r in reports) for k in ['gt_nodes','harmonic_nodes','nucverse_nodes','harmonic_matched','nucverse_matched','nucverse_only','harmonic_only','union_coverage_upper_bound']}
    result=dict(status='complete',seconds=time.monotonic()-start,frames=reports,totals=totals,
        gpu_inference_seconds=field_result['seconds'],scope='Sparse one-to-one node coverage only; union is diagnostic, not a valid prediction or tracking score; unmatched detections are not false positives',
        gpu=False,leaderboard_submitted=False)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('NUCVERSE_EVALUATED',json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
