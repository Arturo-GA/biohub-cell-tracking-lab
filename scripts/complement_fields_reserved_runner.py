"""Frozen dense/static neural fields on the eight reused association videos."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,blosc2
from ensemble_evaluate_runner import one,sha
from biohub_lab.dense_center import DenseCenter
from biohub_lab.temporal_detector import CenterField

def worker(package,rank,world):
    start=time.monotonic();out=Path('/kaggle/working/complement_fields_reserved');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e053_protocol.json').read_text());assert torch.cuda.is_available();torch.set_num_threads(2)
    dp=one('dense_supervision_training/result.json');dm=json.loads(dp.read_text());assert sha(dp)==cfg['dense_manifest_sha256']
    wp=dp.parent/dm['checkpoint'];assert sha(wp)==dm['checkpoint_sha256']
    dense=DenseCenter().cuda().eval();dense.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    sp=one('temporal_detector_training/result.json');sm=json.loads(sp.read_text());assert sha(sp)==cfg['static_manifest_sha256']
    sr=next(r for r in sm['models'] if r['channels']==1);wp=sp.parent/sr['checkpoint'];assert sha(wp)==sr['sha256']
    static=CenterField(1).cuda().eval();static.load_state_dict(torch.load(wp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists());records=[]
    for video in cfg['videos'][rank::world]:
        root=train/(video+'.zarr');q=json.loads((root/'zarr.json').read_text())['attributes']['image_statistics']['quantiles'];low,high=float(q['0.001']),float(q['0.999'])
        meta=json.loads((root/'0/zarr.json').read_text());shape=meta['shape'];arrays={}
        assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*shape[1:]],'Expected one complete volume per chunk'
        for t in range(shape[0]):
            if time.monotonic()-start>1200:raise TimeoutError('Frozen fields GPU budget')
            decoded=np.frombuffer(blosc2.decompress((root/f'0/c/{t}/0/0/0').read_bytes()),dtype=meta['data_type']).reshape(shape[1:])
            raw=np.maximum((np.asarray(decoded[:,::4,::4],np.float32)-low)/(high-low+1e-6),0).astype(np.float16)
            frame=np.clip(raw.astype(np.float32),0,1);x=torch.as_tensor(frame,device='cuda')[None,None]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                values={'logits':dense(x)[0].cpu().numpy().astype(np.float16),'field':static(x)[0].cpu().numpy().astype(np.float16),'image':raw}
            if not arrays:
                for key,v in values.items():arrays[key]=np.lib.format.open_memmap(out/(video+'_'+key+'.npy'),mode='w+',dtype=np.float16,shape=(shape[0],*v.shape))
            for key,v in values.items():arrays[key][t]=v
        for a in arrays.values():a.flush()
        record=dict(video=video,quantiles=[low,high],files={})
        for key in arrays:
            p=out/(video+'_'+key+'.npy');record['files'][key]=dict(file=p.name,sha256=sha(p))
        records.append(record);print('ENSEMBLE_RESERVED_FIELDS',video,flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(records))

def main(package):
    start=time.monotonic();n=min(2,torch.cuda.device_count());assert n
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    codes=[p.wait() for p in jobs];assert all(c==0 for c in codes),codes
    out=Path('/kaggle/working/complement_fields_reserved');records=sum([json.loads((out/f'shard{i}.json').read_text()) for i in range(n)],[])
    cfg=json.loads((package/'baseline/e053_protocol.json').read_text());assert sorted(r['video'] for r in records)==sorted(cfg['videos'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,seconds=time.monotonic()-start,trained=False,annotations_read=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
