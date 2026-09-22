"""GPU primary public association on the expanded, fixed coordinate union."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,blosc2
from residual_detector_io import one,sha
from biohub_lab.independent_expert import load,probabilities,sparse

def worker(package,rank,world):
    start=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    cfg=json.loads((package/'baseline/e063_protocol.json').read_text());pins=json.loads((package/'baseline/e063_infer_pins.json').read_text())
    out=Path('/kaggle/working/selector_associations');out.mkdir(exist_ok=True)
    p=one('selector_inputs/result.json');assert sha(p)==pins['PREPARE'];manifest=json.loads(p.read_text())
    repo=one('repo/scripts/train_unet_transformer.py').parent.parent;weight=one('weights/unet_transformer/split_0/edge_predictor_best.pth')
    model,pos=load(repo,weight,cfg['primary_sha256'],'cuda')
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());records=[]
    for r in manifest['records'][rank::world]:
        video=r['video'];path=p.parent/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as z:coords=z['coords']
        root=train/(video+'.zarr');meta=json.loads((root/'0/zarr.json').read_text());shape=meta['shape'];assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*shape[1:]]
        q=json.loads((root/'zarr.json').read_text())['attributes']['image_statistics']['quantiles'];lo,hi=float(q['0.001']),float(q['0.999']);images=[]
        for t in range(shape[0]):
            raw=np.frombuffer(blosc2.decompress((root/f'0/c/{t}/0/0/0').read_bytes()),dtype=meta['data_type']).reshape(shape[1:])
            images.append(np.maximum((np.asarray(raw[:,::4,::4],np.float32)-lo)/(hi-lo+1e-6),0))
        edges=[];scores=[]
        for start_t in range(0,shape[0]-1,2):
            if time.monotonic()-start>3600:raise TimeoutError('Neural complement GPU budget')
            times=list(range(start_t,min(start_t+2,shape[0]-1)));batch=np.stack([np.stack(images[t:t+2]) for t in times])
            with torch.inference_mode():
                features,_=model.encode(torch.as_tensor(batch,device='cuda'))
                for i,t in enumerate(times):
                    a=np.flatnonzero(coords[:,0]==t);b=np.flatnonzero(coords[:,0]==t+1)
                    prob=probabilities(model,pos,features[i:i+1],coords[a],coords[b]);e,v=sparse(prob,np.empty((0,2),int))
                    edges.append(np.c_[a[e[:,0]],b[e[:,1]]]);scores.append(v)
        path=out/(video+'.npz');np.savez_compressed(path,edges=np.concatenate(edges).astype(np.int32),prob=np.concatenate(scores))
        records.append(dict(video=video,file=path.name,sha256=sha(path),union_sha256=r['sha256'],pairs=shape[0]-1,edges=sum(map(len,edges))))
        print('ENSEMBLE_NEURAL_ASSOCIATION',video,records[-1]['edges'],flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(records))

def main(package):
    start=time.monotonic();n=min(2,torch.cuda.device_count());assert n
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    codes=[p.wait() for p in jobs];assert all(c==0 for c in codes),codes
    out=Path('/kaggle/working/selector_associations');records=sum([json.loads((out/f'shard{i}.json').read_text()) for i in range(n)],[])
    cfg=json.loads((package/'baseline/e063_protocol.json').read_text());assert sorted(r['video'] for r in records)==sorted((cfg['fit']+cfg['development']))
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,annotations_read=False,trained=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
