"""E041 GPU only: genuine dense supervision; development images never fit."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.dense_center import DenseCenter
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/dense_supervision_training');out.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e041_protocol.json').read_text());p=one('dense_supervision/result.json');m=json.loads(p.read_text())
    assert sha(p)==cfg['data_sha256'];data=[]
    for r in m['videos']:
        q=p.parent/r['file'];assert sha(q)==r['sha256']
        with np.load(q) as d:data.append({k:d[k] for k in ['x','y','mask']})
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=DenseCenter().cuda();initial=model.first[0].weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');history=[];updates=0
    for step in range(cfg['steps']):
        if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('E041 GPU deadline')
        d=data[int(rng.integers(len(data)))];ix=rng.integers(len(d['x']),size=8)
        x=d['x'][ix].astype(np.float32);y=d['y'][ix].astype(np.float32);mask=d['mask'][ix]
        for axis in range(1,4):
            if rng.random()<.5:x=np.flip(x,axis);y=np.flip(y,axis);mask=np.flip(mask,axis)
        x=torch.as_tensor((x*(.8+.4*rng.random())).copy(),device='cuda')[:,None];y=torch.as_tensor(y.copy(),device='cuda');mask=torch.as_tensor(mask.copy(),device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(x);loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*(1+20*y)*mask).sum()/((1+20*y)*mask).sum().clamp_min(1)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scale=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=scale)
        if (step+1)%250==0:history.append(dict(step=step+1,loss=float(loss.detach())));print('DENSE_TRAIN',history[-1],flush=True)
    delta=float((model.first[0].weight-initial).norm());assert delta>0
    weight=out/'dense_center.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg),weight)
    p=one('three_lines_data/result.json');manifest=json.loads(p.read_text());assert sha(p)==cfg['biohub_data_sha256'];records=[];model.eval()
    for r in manifest['videos']:
        if r['split']!='development':continue
        q=p.parent/r['image'];assert sha(q)==r['image_sha256'];vol=np.load(q,mmap_mode='r');pred=[]
        for t in cfg['frames']:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(torch.as_tensor(np.array(vol[t:t+1],np.float32),device='cuda')[:,None])
            pred.append(logits.float().cpu().numpy()[0])
        path=out/(r['video']+'.npz');np.savez_compressed(path,logits=np.asarray(pred));records.append(dict(video=r['video'],file=path.name,sha256=sha(path)))
        print('DENSE_INFER',r['video'],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,checkpoint=weight.name,checkpoint_sha256=sha(weight),updates=updates,first_layer_change=delta,history=history,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
