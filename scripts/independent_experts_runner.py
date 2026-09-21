"""E059 GPU only: two standalone checkpoints on fixed detected cells, no GT."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,blosc2
from biohub_lab.independent_expert import load,probabilities,sparse

def sha(path):
    import hashlib
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def find(relative):
    paths=list(dict.fromkeys(p for depth in [1,2,3] for p in Path('/kaggle/input').glob('*/'*depth+relative)));assert len(paths)==1,(relative,paths);return paths[0]

def worker(package,rank,world):
    start=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    out=Path('/kaggle/working/independent_experts');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e059_protocol.json').read_text());repo=find('repo/scripts/train_unet_transformer.py').parent.parent
    capture=find('visual_validation_capture/result.json').parent;assert json.loads((capture/'result.json').read_text())['status']=='complete'
    manifest=json.loads((capture/'capture_manifest.json').read_text())['files'];cache=capture.parent/'visual_candidate_cache'
    train=next(p/'train' for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),Path('/kaggle/input/biohub-cell-tracking-during-development')] if (p/'train').exists())
    models={name:load(repo,find(spec['file']),spec['sha256'],'cuda') for name,spec in cfg['models'].items()};records=[]
    for video in cfg['videos'][rank::world]:
        root=train/(video+'.zarr');meta=json.loads((root/'0/zarr.json').read_text());shape=meta['shape'];assert meta['chunk_grid']['configuration']['chunk_shape']==[1,*shape[1:]]
        q=json.loads((root/'zarr.json').read_text())['attributes']['image_statistics']['quantiles'];low,high=float(q['0.001']),float(q['0.999'])
        images=[]
        for t in range(shape[0]):
            raw=np.frombuffer(blosc2.decompress((root/f'0/c/{t}/0/0/0').read_bytes()),dtype=meta['data_type']).reshape(shape[1:])
            images.append(np.maximum((np.asarray(raw[:,::4,::4],np.float32)-low)/(high-low+1e-6),0))
        frames=[]
        for t in range(shape[0]-1):
            path=cache/video/(str(t)+'.npz');assert sha(path)==manifest[str(path.relative_to(cache))]
            with np.load(path,allow_pickle=False) as z:frames.append({k:z[k] for k in z.files})
        for name,(model,pos) in models.items():
            folder=out/name/video;folder.mkdir(parents=True,exist_ok=True)
            for start_t in range(0,len(frames),2):
                if time.monotonic()-start>1200:raise TimeoutError('Independent experts GPU budget')
                times=list(range(start_t,min(start_t+2,len(frames))));batch=np.stack([np.stack(images[t:t+2]) for t in times])
                with torch.inference_mode():
                    features,_=model.encode(torch.as_tensor(batch,device='cuda'))
                    for i,t in enumerate(times):
                        old=frames[t];prob=probabilities(model,pos,features[i:i+1],old['source_coords'],old['target_coords']);edges,p=sparse(prob,old['edges'])
                        np.savez_compressed(folder/(str(t)+'.npz'),source_coords=old['source_coords'],target_coords=old['target_coords'],edges=edges,prob=p,source_ids=old['source_ids'],target_ids=old['target_ids'])
                del features
            files={p.name:sha(p) for p in sorted(folder.glob('*.npz'))};records.append(dict(video=video,model=name,pairs=len(files),files=files));print('ENSEMBLE_INDEPENDENT_CAPTURE',video,name,len(files),flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(records))

def main(package):
    start=time.monotonic();n=min(2,torch.cuda.device_count());assert n
    jobs=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    codes=[p.wait() for p in jobs];assert all(c==0 for c in codes),codes
    out=Path('/kaggle/working/independent_experts');rows=sum([json.loads((out/f'shard{i}.json').read_text()) for i in range(n)],[]);cfg=json.loads((package/'baseline/e059_protocol.json').read_text())
    assert {(r['video'],r['model']) for r in rows}=={(v,m) for v in cfg['videos'] for m in cfg['models']}
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=rows,seconds=time.monotonic()-start,annotations_read=False,trained=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
