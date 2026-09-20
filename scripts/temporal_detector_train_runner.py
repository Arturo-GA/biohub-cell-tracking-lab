"""E039 GPU: paired single-frame / three-frame fully convolutional center fields."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.temporal_detector import CenterField,vector_targets
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/temporal_detector_training');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e039_protocol.json').read_text())
    p=one('three_lines_data/result.json');manifest=json.loads(p.read_text());assert sha(p)==cfg['data_manifest_sha256'];data=[]
    for r in manifest['videos']:
        if r['split']!='fit':continue
        q=p.parent/r['truth'];assert sha(q)==r['truth_sha256']
        with np.load(q) as d:coords=d['coords']/[1,1,4,4]
        v=p.parent/r['image'];assert sha(v)==r['image_sha256'];data.append((r['video'],np.load(v,mmap_mode='r'),coords))
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;records=[]
    for channels in [1,3]:
        torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);model=CenterField(channels).cuda();initial=model.first[0].weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');losses=[];tick=time.monotonic();updates=0
        for step in range(cfg['steps']):
            if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('GPU budget reached')
            xs=[];ys=[];ms=[]
            for _ in range(cfg['batch']):
                _,vol,coords=data[int(rng.integers(len(data)))];n=int(rng.integers(len(coords)));t=int(coords[n,0]);center=np.rint(coords[n,1:]+rng.uniform(-8,8,3)).astype(int);lo=center-16
                frames=np.clip(np.arange(t-1,t+2),0,len(vol)-1);volume=np.array(vol[frames],np.float32);volume=np.pad(volume,((0,0),(32,32),(32,32),(32,32)),mode='edge')
                sl=tuple(slice(int(a+32),int(a+64)) for a in lo);x=volume[(slice(None),*sl)];assert x.shape==(3,32,32,32)
                points=coords[coords[:,0]==t,1:]-lo;target,mask=vector_targets(points)
                assert mask.any()
                for axis in range(3):
                    if rng.random()<.5:x=np.flip(x,axis+1);target=np.flip(target,axis+1).copy();mask=np.flip(mask,axis).copy();target[axis]*=-1
                x=x*(.8+.4*rng.random());xs.append(x if channels==3 else x[1:2]);ys.append(target);ms.append(mask)
            x=torch.as_tensor(np.stack(xs),device='cuda');y=torch.as_tensor(np.stack(ys),device='cuda');mask=torch.as_tensor(np.stack(ms),device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x);loss=(F.smooth_l1_loss(pred,y,reduction='none').sum(1)*mask).sum()/(3*mask.sum())
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=previous)
            if (step+1)%250==0:losses.append(dict(step=step+1,loss=float(loss.detach())));print('CENTER_TRAIN',channels,losses[-1],flush=True)
        delta=float((model.first[0].weight.detach()-initial).norm());assert delta>0
        path=out/f'center_{channels}.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},channels=channels,config=cfg),path)
        records.append(dict(channels=channels,checkpoint=path.name,sha256=sha(path),updates=updates,first_layer_change=delta,history=losses,seconds=time.monotonic()-tick))
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,models=records,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
