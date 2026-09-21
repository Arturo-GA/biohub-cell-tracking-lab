"""E052 GPU-only frozen DeepCenter TTA maps, two independent shards."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from ensemble_evaluate_runner import sha
from biohub_lab.public_postprocess import load

def worker(package,rank,world):
    start=time.monotonic();out=Path('/kaggle/working/public_division_fields');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e052_protocol.json').read_text());names=cfg['videos'][rank::world]
    module=load((package/'baseline/harmonic_inference.py').read_text());module.TEST_DIR=module.COMP_DIR/'train'
    candidates=[Path('/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt'),Path('/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')]
    wp=next(p for p in candidates if p.is_file());assert sha(wp)==cfg['checkpoint_sha256']
    module._dc_checkpoint_candidates=lambda:[wp]
    torch.set_num_threads(2);assert torch.cuda.is_available();bundle=module.load_deepcenter_veto_detector();records=[]
    for name in names:
        shape=json.loads((module.TEST_DIR/(name+'.zarr')/'0/zarr.json').read_text())['shape'];arr=None;path=out/(name+'.npy')
        for t in range(shape[0]):
            if time.monotonic()-start>1800:raise TimeoutError('DeepCenter shard process budget')
            heat=module.deepcenter_heatmap_for_frame(name,t,bundle,{},{}).astype(np.float32)
            if arr is None:arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(shape[0],*heat.shape))
            arr[t]=heat
        arr.flush();del arr
        records.append(dict(video=name,file=path.name,sha256=sha(path),frames=shape[0]));print('ENSEMBLE_DIVISION_FIELDS',name,flush=True)
    (out/f'shard{rank}.json').write_text(json.dumps(dict(videos=records,pool_factor=int(getattr(bundle['cfg'],'pool_factor',4)),seconds=time.monotonic()-start)))

def main(package):
    n=min(2,torch.cuda.device_count());assert n
    start=time.monotonic();processes=[subprocess.Popen([sys.executable,'-u',__file__,str(package),str(i),str(n)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2')) for i in range(n)]
    codes=[p.wait() for p in processes];assert all(c==0 for c in codes),codes
    out=Path('/kaggle/working/public_division_fields');records=[]
    for i in range(n):records+=json.loads((out/f'shard{i}.json').read_text())['videos']
    cfg=json.loads((package/'baseline/e052_protocol.json').read_text());assert sorted(r['video'] for r in records)==sorted(cfg['videos'])
    pool_factors={json.loads((out/f'shard{i}.json').read_text())['pool_factor'] for i in range(n)};assert len(pool_factors)==1
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,pool_factor=pool_factors.pop(),videos=records,seconds=time.monotonic()-start,trained=False),indent=2))
if __name__=='__main__':
    if len(sys.argv)==4:worker(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
