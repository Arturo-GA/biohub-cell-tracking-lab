import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from biohub_lab.temporal_volume import TemporalVolumeEncoder

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def locate(name):
    root=Path('/kaggle/input');parents=[root,*root.glob('*'),*root.glob('*/*'),*root.glob('*/*/*')];found=list(dict.fromkeys(p/name for p in parents if (p/name).is_file()));assert len(found)==1;return found[0]
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/neural_division_features');root.mkdir(exist_ok=True);torch.set_num_threads(2);assert torch.cuda.is_available()
    config=json.loads((package/'baseline/e037_neural_division.json').read_text());p=locate('neural_division_data/result.json');data=json.loads(p.read_text());assert data['status']=='complete' and data['config']==config
    tp=locate('temporal_volume_training/result.json');assert sha(tp)==config['encoder_manifest_sha256'];models={};folds=json.loads(tp.read_text())['folds']
    for f in folds:
        assert not set(f['training_videos'])&{r['video'] for r in data['videos']}
        checkpoint=tp.parent/f['checkpoint'];assert sha(checkpoint)==f['checkpoint_sha256'];saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=TemporalVolumeEncoder().cuda().eval();net.load_state_dict(saved['state_dict'],strict=True);models[f['heldout_embryo']]=net
    records=[]
    for item in data['videos']:
        cp=p.parent/item['crops'];assert sha(cp)==item['crops_sha256'];crops=np.load(cp,mmap_mode='r');values={}
        for fold,net in models.items():
            parts=[]
            with torch.inference_mode():
                for first in range(0,len(crops),128):
                    if time.monotonic()-start>600:raise TimeoutError('Neural inference budget exceeded')
                    x=torch.from_numpy(np.array(crops[first:first+128],np.float32)).cuda()
                    with torch.autocast('cuda',dtype=torch.float16):h=net(x)
                    parts.append(h.float().cpu().numpy())
            values[fold]=np.concatenate(parts) if parts else np.empty((0,64),np.float32)
            assert np.isfinite(values[fold]).all()
        target=root/(item['video']+'_features.npz');np.savez_compressed(target,**values)
        records.append(dict(video=item['video'],file=target.name,sha256=sha(target),crops_sha256=item['crops_sha256']));print('NEURAL_FEATURES',item['video'],len(crops),flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,outputs=records,input_manifest_sha256=sha(p),seconds=time.monotonic()-start,training=False,annotations_read=False),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
