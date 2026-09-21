"""E047 only frozen dense+static neural inference; no optimizer or graph processing."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from ensemble_evaluate_runner import one,sha
from biohub_lab.dense_center import DenseCenter
from biohub_lab.temporal_detector import CenterField

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e047_protocol.json').read_text());out=Path('/kaggle/working/consensus_fields');out.mkdir(exist_ok=True)
    p=one('dense_detector_inputs/result.json');assert sha(p)==cfg['raw_sha256'];raw=json.loads(p.read_text())
    d=one('dense_supervision_training/result.json');assert sha(d)==cfg['dense_sha256'];dm=json.loads(d.read_text())
    s=one('temporal_detector_training/result.json');assert sha(s)==cfg['static_sha256'];sm=json.loads(s.read_text())
    assert torch.cuda.is_available();torch.set_num_threads(2)
    dense=DenseCenter().cuda().eval();wp=d.parent/dm['checkpoint'];assert sha(wp)==dm['checkpoint_sha256'];dense.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    sr=next(r for r in sm['models'] if r['channels']==1);wp=s.parent/sr['checkpoint'];assert sha(wp)==sr['sha256'];static=CenterField(1).cuda().eval();static.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    records=[]
    for r in raw['videos']:
        if time.monotonic()-start>1200:raise TimeoutError('E047 GPU process budget')
        ip=p.parent/r['file'];assert sha(ip)==r['sha256'];image=np.load(ip,mmap_mode='r');shape=image.shape
        lp=out/(r['video']+'_logits.npy');fp=out/(r['video']+'_field.npy')
        logits=np.lib.format.open_memmap(lp,mode='w+',dtype=np.float16,shape=shape);fields=np.lib.format.open_memmap(fp,mode='w+',dtype=np.float16,shape=(shape[0],3,*shape[1:]))
        for i in range(0,len(image),4):
            x=torch.as_tensor(np.clip(np.asarray(image[i:i+4],np.float32),0,1),device='cuda')[:,None]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):a=dense(x);b=static(x)
            logits[i:i+len(x)]=a.cpu().numpy();fields[i:i+len(x)]=b.cpu().numpy()
        logits.flush();fields.flush();del logits,fields
        records.append(dict(video=r['video'],logits=lp.name,logits_sha256=sha(lp),field=fp.name,field_sha256=sha(fp),frames=len(image)))
        print('ENSEMBLE_INFER',r['video'],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,seconds=time.monotonic()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,trained=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
