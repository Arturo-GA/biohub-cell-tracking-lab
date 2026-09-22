"""Actual neural inference on fit/development movies; no annotation access."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,blosc2
from torch.nn import functional as F
from residual_detector_io import one,sha
from biohub_lab.residual_detector import TemporalDenseCenter
from biohub_lab.independent_expert import load

def peaks(logits,threshold,topk=512):
    p=logits.float().sigmoid().reshape(1,1,*logits.shape[-3:])
    flat=torch.where((p==F.max_pool3d(p,3,1,1))&(p>=threshold),p,0).flatten()
    v,ix=torch.topk(flat,min(topk,len(flat)));ix=ix[v>0];v=v[v>0]
    z,y,x=p.shape[-3:]
    return torch.stack([ix//(y*x),(ix//x)%y,ix%x],1).cpu().numpy()*[1,4,4],v.cpu().numpy()

def worker(package,rank,world):
    start=time.monotonic();torch.set_num_threads(2)
    cfg=json.loads((package/'baseline/e063_protocol.json').read_text());out=Path('/kaggle/working/selector_detections');out.mkdir(exist_ok=True)
    p=one('residual_detector_training/result.json');assert sha(p)==cfg['training_sha256'];m=json.loads(p.read_text());w=p.parent/m['checkpoint'];assert sha(w)==m['checkpoint_sha256']
    adapted=TemporalDenseCenter().cuda().eval();adapted.load_state_dict(torch.load(w,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    repo=one('repo/scripts/train_unet_transformer.py').parent.parent
    primary,_=load(repo,one('weights/unet_transformer/split_0/edge_predictor_best.pth'),cfg['primary_sha256'],'cuda')
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());records=[]
    for video in (cfg['fit']+cfg['development'])[rank::world]:
        root=train/(video+'.zarr');meta=json.loads((root/'0/zarr.json').read_text());shape=meta['shape'];assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*shape[1:]]
        q=json.loads((root/'zarr.json').read_text())['attributes']['image_statistics']['quantiles'];lo,hi=float(q['0.001']),float(q['0.999']);images=[]
        for t in range(shape[0]):
            raw=np.frombuffer(blosc2.decompress((root/f'0/c/{t}/0/0/0').read_bytes()),dtype=meta['data_type']).reshape(shape[1:])
            images.append(np.maximum((np.asarray(raw[:,::4,::4],np.float32)-lo)/max(hi-lo,1e-6),0))
        coords={a:[] for a in ('primary','adapted')};prob={a:[] for a in coords}
        with torch.inference_mode():
            for t in range(0,len(images),2):
                if time.monotonic()-start>3600:raise TimeoutError('E063 neural inference budget')
                times=list(range(t,min(t+2,len(images))));pair=torch.as_tensor(np.stack([images[i] for i in times]),device='cuda')[None]
                _,det=primary.encode(pair)
                for j,i in enumerate(times):
                    ijk,p=peaks(det[j],.96875);coords['primary'].append(np.c_[np.full(len(p),i),ijk]);prob['primary'].append(p)
                    x=torch.as_tensor(np.clip(np.stack([images[max(0,i-1)],images[i],images[min(i+1,len(images)-1)]]),0,1),device='cuda')[None]
                    with torch.autocast('cuda',dtype=torch.float16):logits=adapted(x)
                    ijk,p=peaks(logits,.2);coords['adapted'].append(np.c_[np.full(len(p),i),ijk]);prob['adapted'].append(p)
        path=out/(video+'.npz');np.savez_compressed(path,**{a+'_coords':np.concatenate(coords[a]).astype(np.int32) for a in coords},**{a+'_prob':np.concatenate(prob[a]) for a in prob})
        records.append(dict(video=video,file=path.name,sha256=sha(path),shape=shape));print('ENSEMBLE_SELECTOR_DETECT',video,flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(records))

def main(package):
    start=time.monotonic();n=min(2,torch.cuda.device_count());assert n
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    assert all(c==0 for c in [p.wait() for p in jobs])
    out=Path('/kaggle/working/selector_detections');records=sum([json.loads((out/f'shard{i}.json').read_text()) for i in range(n)],[])
    cfg=json.loads((package/'baseline/e063_protocol.json').read_text());assert sorted(r['video'] for r in records)==sorted(cfg['fit']+cfg['development'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,annotations_read=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
