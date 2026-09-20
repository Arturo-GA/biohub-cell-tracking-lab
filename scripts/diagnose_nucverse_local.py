"""Local CPU failure localization; does not retune or generate predictions."""
import hashlib,json
import numpy as np
from scipy import ndimage as ndi
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'outputs/e024_diagnostic_data'
    manifest=json.loads((ROOT/'outputs/e024_gpu_recovery/nucverse_fields/result.json').read_text())
    records=[]
    for frame in manifest['frames']:
        path=root/frame['output'];assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['output_sha256']
        fields=np.load(path,allow_pickle=False);prob=np.asarray(fields[...,0],np.float32)
        with np.load(root/(Path(frame['file']).stem+'_reference.npz'),allow_pickle=False) as ref:gt=ref['truth']
        probability=ndi.map_coordinates(prob,gt.T,order=1,mode='nearest')
        distance=ndi.distance_transform_edt(prob<=.5,sampling=(1.625,.40625,.40625))
        nearest=ndi.map_coordinates(distance,gt.T,order=1,mode='nearest')
        records.append(dict(video=frame['video'],frame=frame['frame'],gt_nodes=len(gt),
            foreground_at_gt=int((probability>.5).sum()),foreground_within_1_5um=int((nearest<=1.5).sum()),foreground_within_7um=int((nearest<=7).sum()),
            median_probability=float(np.median(probability)) if len(gt) else None))
    totals={k:sum(r[k] for r in records) for k in ['gt_nodes','foreground_at_gt','foreground_within_1_5um','foreground_within_7um']}
    result=dict(status='complete',frames=records,totals=totals,gpu_used=False,thresholds_retuned=False,
        scope='Post hoc localization of mask-vs-instance failure on the same 39 points, not a new validation set')
    (ROOT/'results/E024_mask_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
