import hashlib,json,time
from pathlib import Path
import numpy as np
from dense_temporal_rank_runner import match
from biohub_lab.temporal_volume import records,crop_temporal
from association_cpu_runner import locate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/temporal_volume_data');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e033_temporal_volume.json').read_text())
    source=locate(inputs,'dense_detector_predictions/result.json');raw=locate(inputs,'dense_detector_inputs/result.json')
    assert sha(source)==config['prediction_manifest_sha256'] and sha(raw)==config['image_manifest_sha256']
    pred=json.loads(source.read_text());images=json.loads(raw.read_text());report=[]
    for item in pred['files']:
        if item['mode']!='batch_bn':continue
        p=source.parent/item['file'];assert sha(p)==item['sha256']
        with np.load(p,allow_pickle=False) as d:coords=d['coords']
        truth=json.loads((raw.parent/(item['video']+'_truth.json')).read_text());g=np.asarray(truth['nodes'],float)
        data=records(coords,g,truth['edges'],match(coords,g));assert len(data['labels'])
        unique,inverse=np.unique(data['indices'],return_inverse=True);data['indices']=inverse.reshape(-1,13)
        ir=next(r for r in images['videos'] if r['video']==item['video']);ip=raw.parent/ir['file'];assert sha(ip)==ir['sha256']
        image=np.load(ip,allow_pickle=False);assert np.isfinite(image).all()
        padded=np.pad(np.log1p(image.astype(np.float32)),((0,0),(8,8),(8,8),(8,8)),mode='edge')
        path=root/(item['video']+'_crops.npy');crops=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(unique),3,16,16,16))
        for i,n in enumerate(unique):crops[i]=crop_temporal(padded,coords[n])
        crops.flush();del crops,padded,image
        metadata=root/(item['video']+'_candidates.npz');np.savez_compressed(metadata,**data)
        record=dict(video=item['video'],crops=path.name,crops_sha256=sha(path),candidates=metadata.name,candidates_sha256=sha(metadata),targets=len(data['labels']),unique_crops=len(unique),train_competitive=int((data['mask'].sum(1)>1).sum()),nearest_correct=int((np.where(data['valid'],data['distance'],np.inf).argmin(1)==data['labels']).sum()))
        report.append(record);print('VOLUME_DATA',json.dumps(record),flush=True)
    result=dict(status='complete',config=config,videos=report,seconds=time.monotonic()-start,selection_uses_model_errors=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print('VOLUME_DATA_COMPLETE',flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
