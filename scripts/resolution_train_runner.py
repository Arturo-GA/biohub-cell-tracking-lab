"""E042 identical examples, architecture and seed; native vs XY4 information."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from biohub_lab.resolution_probe import ResolutionProbe,coarse_control
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/resolution_training');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e042_protocol.json').read_text());p=one('resolution_data/result.json');m=json.loads(p.read_text());assert sha(p)==cfg['data_sha256']
    data={}
    for r in m['videos']:
        q=p.parent/r['file'];assert sha(q)==r['sha256']
        with np.load(q) as d:data[r['video']]=(d['high'],d['offset'],r['split'])
    fit=[v for v in data.values() if v[2]=='fit'];torch.set_num_threads(2);assert torch.cuda.is_available();records=[]
    for arm in ['coarse','native']:
        torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);model=ResolutionProbe().cuda();initial=model.net[0].weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');updates=0;tick=time.monotonic();history=[]
        for step in range(cfg['steps']):
            if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('E042 GPU budget')
            v=fit[int(rng.integers(len(fit)))];ix=rng.integers(len(v[0]),size=16);x=torch.as_tensor(v[0][ix].astype(np.float32),device='cuda');y=torch.as_tensor(v[1][ix],device='cuda')
            if arm=='coarse':x=coarse_control(x)
            # No flips: avoid half-voxel center changes as a confound in this paired audit.
            x=x*(.8+.4*rng.random())
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.smooth_l1_loss(model(x),y)
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scale=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=scale)
            if (step+1)%250==0:history.append(dict(step=step+1,loss=float(loss.detach())));print('RESOLUTION_TRAIN',arm,history[-1],flush=True)
        delta=float((model.net[0].weight.detach()-initial).norm());assert delta>0 and updates>=.9*cfg['steps']
        weight=out/(arm+'.pt');torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,arm=arm),weight);model.eval();outputs=[]
        for name,(x,y,split) in data.items():
            if split!='development':continue
            predictions=[];shuffled_predictions=[]
            for first in range(0,len(x),32):
                t=torch.as_tensor(x[first:first+32].astype(np.float32),device='cuda')
                if arm=='coarse':t=coarse_control(t)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    predictions.append(model(t).float().cpu().numpy());shuffled=t.clone();shuffled[:,1]=torch.roll(t[:,1],1,0);shuffled_predictions.append(model(shuffled).float().cpu().numpy())
            path=out/(name+'_'+arm+'.npz');np.savez_compressed(path,pred=np.concatenate(predictions),pred_shuffled=np.concatenate(shuffled_predictions),target=y);outputs.append(dict(video=name,file=path.name,sha256=sha(path)))
        records.append(dict(arm=arm,checkpoint=weight.name,sha256=sha(weight),updates=updates,first_layer_change=delta,seconds=time.monotonic()-tick,history=history,outputs=outputs))
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,models=records,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
