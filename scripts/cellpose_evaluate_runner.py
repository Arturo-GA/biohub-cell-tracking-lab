"""E028 CPU: native Cellpose mask decoder and paired sparse coverage."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from cellpose_runtime import setup,locate
from biohub_lab.nucverse_instances import matched_truth
def main(package):
    root=Path('/kaggle/working/cellpose_evaluation');root.mkdir(exist_ok=True);start=time.monotonic();setup(package)
    import torch
    assert not torch.cuda.is_available()
    from cellpose import dynamics
    source=locate('cellpose_fields/result.json');report=json.loads(source.read_text());assert report['status']=='complete'
    frozen=[]
    for item in report['frames']:
        path=source.parent/item['output'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['output_sha256']
        with np.load(path,allow_pickle=False) as d:mask=dynamics.compute_masks(d['dP'],d['cellprob'],niter=200,cellprob_threshold=0.,do_3D=True,min_size=15,max_size_fraction=.4,device=torch.device('cpu'))
        labels=np.unique(mask);labels=labels[labels>0]
        centers=np.asarray(ndi.center_of_mass(np.ones(mask.shape,np.uint8),mask,labels),float).reshape(-1,3) if len(labels) else np.empty((0,3))
        target=root/(Path(item['file']).stem+'_centers.npy');np.save(target,centers,allow_pickle=False)
        frozen.append(dict(frame=item,prediction=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()));print('CELLPOSE_DECODED',item['video'],item['frame'],len(centers),flush=True)
    (root/'frozen_predictions.json').write_text(json.dumps(dict(annotations_read=False,frames=frozen),indent=2)+'\n')
    references=locate('nucverse_inputs/result.json').parent;rows=[]
    for row in frozen:
        frame=row['frame'];prediction=np.load(root/row['prediction'],allow_pickle=False)
        with np.load(references/(Path(frame['file']).stem+'_reference.npz'),allow_pickle=False) as d:truth=d['truth'];harmonic=d['harmonic']
        old=matched_truth(harmonic,truth);new=matched_truth(prediction,truth)
        rows.append(dict(video=frame['video'],frame=frame['frame'],gt_nodes=len(truth),harmonic_nodes=len(harmonic),candidate_nodes=len(prediction),harmonic_matched=len(old),candidate_matched=len(new),candidate_only=len(new-old)))
    totals={key:sum(r[key] for r in rows) for key in ['gt_nodes','harmonic_nodes','candidate_nodes','harmonic_matched','candidate_matched','candidate_only']}
    result=dict(status='complete',frames=rows,totals=totals,seconds=time.monotonic()-start,gpu=False,leaderboard_submitted=False,scope='Two preselected error-conditioned calibration frames; detection only, not tracking score or an unbiased model comparison')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('CELLPOSE_EVALUATED',json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
