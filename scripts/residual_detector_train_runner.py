"""GPU: positive-only Biohub adaptation with dense external replay and teacher preservation."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.residual_detector import TemporalDenseCenter,DenseCenter
from residual_detector_io import one,sha

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e061_protocol.json').read_text())
    pins=json.loads((package/'baseline/e061_train_pins.json').read_text())
    out=Path('/kaggle/working/residual_detector_training');out.mkdir(exist_ok=True)
    p=one('residual_detector_data/result.json');assert sha(p)==pins['prepared_sha256'];prepared=json.loads(p.read_text());real=[]
    assert set(prepared['config']['fit'])==set(cfg['fit']) and not set(cfg['fit'])&set(cfg['evaluation'])
    for r in prepared['records']:
        path=p.parent/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as z:real.append({k:z[k] for k in ('x','y')})
    p=one('dense_supervision/result.json');assert sha(p)==cfg['external_data_sha256'];external=[]
    for r in json.loads(p.read_text())['videos']:
        path=p.parent/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as z:external.append({k:z[k] for k in ('x','y','mask')})
    p=one('dense_supervision_training/result.json');assert sha(p)==cfg['pretrained_manifest_sha256'];m=json.loads(p.read_text())
    path=p.parent/m['checkpoint'];assert sha(path)==m['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    teacher=DenseCenter().cuda().eval();teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    model=TemporalDenseCenter().cuda();model.initialize(state);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01);scaler=torch.amp.GradScaler('cuda');history=[];updates=0
    for step in range(cfg['steps']):
        if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('E061 training budget')
        rx=[];ry=[];ex=[];ey=[];em=[]
        for _ in range(cfg['batch_real']):
            d=real[int(rng.integers(len(real)))];i=int(rng.integers(len(d['x'])));rx.append(d['x'][i]);ry.append(d['y'][i])
        for _ in range(cfg['batch_external']):
            d=external[int(rng.integers(len(external)))];i=int(rng.integers(len(d['x'])));ex.append(d['x'][i]);ey.append(d['y'][i]);em.append(d['mask'][i])
        rx=np.stack(rx).astype(np.float32);ry=np.stack(ry).astype(np.float32);ex=np.stack(ex).astype(np.float32);ey=np.stack(ey).astype(np.float32);em=np.stack(em)
        for axis in range(3):
            if rng.random()<.5:rx=np.flip(rx,axis+2);ry=np.flip(ry,axis+1);ex=np.flip(ex,axis+1);ey=np.flip(ey,axis+1);em=np.flip(em,axis+1)
        rx=rx*(.8+.4*rng.random());ex=ex*(.8+.4*rng.random())
        x=torch.as_tensor(rx.copy(),device='cuda');y=torch.as_tensor(ry.copy(),device='cuda')
        a=torch.as_tensor(ex.copy(),device='cuda')[:,None].repeat(1,3,1,1,1);b=torch.as_tensor(ey.copy(),device='cuda');mask=torch.as_tensor(em.copy(),device='cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):reference=teacher(x[:,1:2]).sigmoid()
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(x);known=y>.3;hard=(1+3*(y-reference).clamp_min(0))*known
            positive=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*hard).sum()/hard.sum().clamp_min(1)
            # Soft preservation is a teacher prior, not a ground-truth background label.
            outside=y<.01;preserve=((logits.sigmoid()-reference).square()*outside).sum()/outside.sum().clamp_min(1)
            replay_logits=model(a);weight=(1+20*b)*mask
            replay=(F.binary_cross_entropy_with_logits(replay_logits,b,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
            loss=positive+.1*preserve+replay
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scale=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=scale)
        if (step+1)%250==0:
            history.append(dict(step=step+1,positive=float(positive.detach()),preserve=float(preserve.detach()),replay=float(replay.detach()),loss=float(loss.detach())))
            print('ENSEMBLE_RESIDUAL_TRAIN',history[-1],flush=True)
    change=float(sum((v-initial[k]).float().square().sum() for k,v in model.state_dict().items()).sqrt());assert change>0 and updates>=cfg['steps']*.99
    path=out/'temporal_dense.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg),path)
    result=dict(status='complete',config=cfg,checkpoint=path.name,checkpoint_sha256=sha(path),history=history,updates=updates,weight_change_l2=change,
        side_weight_norm=float(model.first[0].weight[:,[0,2]].norm()),seconds=time.monotonic()-start,trained=True,evaluation_images_read=False,evaluation_labels_read=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
