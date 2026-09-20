"""E039 GPU inference only. Voting and sparse annotation metrics are deferred to CPU."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from biohub_lab.temporal_detector import CenterField
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/temporal_detector_fields');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e039_protocol.json').read_text())
    p=one('three_lines_data/result.json');assert sha(p)==cfg['data_manifest_sha256'];manifest=json.loads(p.read_text());q=one('temporal_detector_training/result.json');trained=json.loads(q.read_text());assert trained['config']==cfg
    assert torch.cuda.is_available();torch.set_num_threads(2);models={};records=[]
    for r in trained['models']:
        weight=q.parent/r['checkpoint'];assert sha(weight)==r['sha256'];d=torch.load(weight,weights_only=True,map_location='cpu');m=CenterField(r['channels']).cuda();m.load_state_dict(d['state_dict'],strict=True);m.eval();models[r['channels']]=m
    for item in manifest['videos']:
        if item['split']!='development':continue
        image=p.parent/item['image'];assert sha(image)==item['image_sha256'];vol=np.load(image,mmap_mode='r');fields={c:[] for c in models}
        for t in cfg['frames']:
            x=torch.as_tensor(np.array(vol[np.clip(np.arange(t-1,t+2),0,len(vol)-1)],np.float32)[None],device='cuda')
            for c,m in models.items():
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):y=m(x if c==3 else x[:,1:2])
                fields[c].append(y[0].float().cpu().numpy().astype(np.float16))
        target=out/(item['video']+'.npz');np.savez_compressed(target,**{f'field_{c}':np.stack(a) for c,a in fields.items()})
        records.append(dict(video=item['video'],file=target.name,sha256=sha(target)));print('CENTER_INFER',item['video'],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,seconds=time.monotonic()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
