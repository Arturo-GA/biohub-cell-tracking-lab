"""GPU neural inference of frozen/adapted detectors on the fixed 24 whole movies."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,blosc2
from torch.nn import functional as F
from biohub_lab.residual_detector import TemporalDenseCenter,DenseCenter
from residual_detector_io import one,sha

def worker(package,rank,world):
    start=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    cfg=json.loads((package/'baseline/e061_protocol.json').read_text());pins=json.loads((package/'baseline/e061_infer_pins.json').read_text())
    out=Path('/kaggle/working/residual_detector_predictions');out.mkdir(exist_ok=True)
    p=one('dense_supervision_training/result.json');assert sha(p)==cfg['pretrained_manifest_sha256'];m=json.loads(p.read_text());path=p.parent/m['checkpoint'];assert sha(path)==m['checkpoint_sha256']
    frozen=DenseCenter().cuda().eval();frozen.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    p=one('residual_detector_training/result.json');assert sha(p)==pins['training_sha256'];m=json.loads(p.read_text());path=p.parent/m['checkpoint'];assert sha(path)==m['checkpoint_sha256']
    adapted=TemporalDenseCenter().cuda().eval();adapted.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());records=[]
    for video in cfg['evaluation'][rank::world]:
        root=train/(video+'.zarr');meta=json.loads((root/'0/zarr.json').read_text());shape=meta['shape'];assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*shape[1:]]
        q=json.loads((root/'zarr.json').read_text())['attributes']['image_statistics']['quantiles'];lo,hi=float(q['0.001']),float(q['0.999']);images=[]
        for t in range(shape[0]):
            raw=np.frombuffer(blosc2.decompress((root/f'0/c/{t}/0/0/0').read_bytes()),dtype=meta['data_type']).reshape(shape[1:])
            images.append(np.clip((np.asarray(raw[:,::4,::4],np.float32)-lo)/max(hi-lo,1e-6),0,1))
        coords={a:[] for a in ('frozen','adapted','repeated')};probs={a:[] for a in coords};features={a:[] for a in coords}
        for t in range(len(images)):
            if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('E061 inference budget')
            x=torch.as_tensor(np.stack([images[max(0,t-1)],images[t],images[min(t+1,len(images)-1)]]),device='cuda')[None]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                predictions={'frozen':frozen(x[:,1:2]),'adapted':adapted(x),'repeated':adapted(x[:,1:2].repeat(1,3,1,1,1))}
                for arm,logits in predictions.items():
                    p=logits.float().sigmoid();peak=(p==F.max_pool3d(p[:,None],3,1,1)[:,0])&(p>=cfg['proposals']['threshold'])
                    flat=torch.where(peak,p,torch.zeros_like(p)).flatten();v,ix=torch.topk(flat,min(cfg['proposals']['topk'],flat.numel()));ix=ix[v>0];v=v[v>0]
                    z,y,w=p.shape[-3:];ijk=torch.stack([ix//(y*w),(ix//w)%y,ix%w],1).cpu().numpy()*[1,4,4]
                    coords[arm].append(np.c_[np.full(len(ijk),t),ijk]);probs[arm].append(v.cpu().numpy())
                    offsets=np.stack(np.meshgrid(*[np.arange(-1,2)]*3,indexing='ij'),-1).reshape(-1,3)
                    at=ijk[:,None,:]//[1,4,4]+offsets[None]
                    at=np.clip(at,0,np.asarray(images[t].shape)-1)
                    patch=images[t][at[:,:,0],at[:,:,1],at[:,:,2]];patch=patch-patch.mean(1,keepdims=True)
                    patch/=np.maximum(np.linalg.norm(patch,axis=1,keepdims=True),1e-6);features[arm].append(patch.astype(np.float16))
        for arm in coords:
            path=out/(video+'_'+arm+'.npz');np.savez_compressed(path,coords=np.concatenate(coords[arm]).astype(np.int32),prob=np.concatenate(probs[arm]),features=np.concatenate(features[arm]))
            records.append(dict(video=video,arm=arm,file=path.name,sha256=sha(path),shape=shape,nodes=sum(map(len,coords[arm]))))
        print('ENSEMBLE_RESIDUAL_INFER',video,{a:sum(map(len,c)) for a,c in coords.items()},flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(records))

def main(package):
    start=time.monotonic();n=min(2,torch.cuda.device_count());assert n
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    codes=[p.wait() for p in jobs];assert all(c==0 for c in codes),codes
    out=Path('/kaggle/working/residual_detector_predictions');records=sum([json.loads((out/f'shard{i}.json').read_text()) for i in range(n)],[])
    cfg=json.loads((package/'baseline/e061_protocol.json').read_text());assert len(records)==len(cfg['evaluation'])*3
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,evaluation_labels_read=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
