"""E040 GPU joint raw-image event training; development scores only, no labels read."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.expanded_events import JointEventModel,query_masks
from spatial_probe_features_runner import one,sha

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/expanded_event_training');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e040_protocol.json').read_text());p=one('expanded_event_data/result.json');manifest=json.loads(p.read_text());assert manifest['config']==cfg
    pin=json.loads((package/'baseline/e040_input_pin.json').read_text());assert sha(p)==pin['manifest_sha256']
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);torch.backends.cudnn.benchmark=True
    data=[];positive=[];negative=[]
    for r in manifest['videos']:
        if r['empty'] or r['split']!='fit':continue
        q=p.parent/r['events'];assert sha(q)==r['events_sha256'];v=p.parent/r['crops'];assert sha(v)==r['crops_sha256']
        with np.load(q) as d:a={k:d[k] for k in ['mother','delta','geometry','y']}
        a['crops']=np.load(v,mmap_mode='r');idx=len(data);data.append(a)
        positive.extend((idx,int(i)) for i in np.flatnonzero(a['y']==1));negative.extend((idx,int(i)) for i in np.flatnonzero(a['y']==0))
    assert len(positive)>=100,'Expanded supervision gate failed';assert len(negative)>1000
    model=JointEventModel().cuda();initial=model.encoder[0].weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');history=[];updates=0
    for step in range(cfg['steps']):
        if time.monotonic()-start>cfg['deadline_seconds']:raise TimeoutError('GPU budget reached')
        chosen=[positive[int(rng.integers(len(positive)))] for _ in range(cfg['batch']//2)]+[negative[int(rng.integers(len(negative)))] for _ in range(cfg['batch']//2)]
        xs=[];ds=[];gs=[];ys=[];mother_offsets=[]
        for n,j in chosen:
            a=data[n];x=np.array(a['crops'][a['mother'][j]],np.float32);delta=a['delta'][j].copy()
            # Spatial reflection about voxel16 on a32cube requires indexed reflection
            # about15.5 and updating query positions by -delta-1.
            mother_offset=np.zeros(3,np.float32)
            for axis in range(3):
                if rng.random()<.5:x=np.flip(x,axis+1);delta[:,axis]=-delta[:,axis]-1;mother_offset[axis]=-1
            mother_offsets.append(mother_offset)
            xs.append(x*(.8+.4*rng.random()));ds.append(delta);gs.append(a['geometry'][j]);ys.append(a['y'][j])
        x=torch.as_tensor(np.stack(xs),device='cuda');d=np.stack(ds);x=torch.cat([x,query_masks(d,'cuda',np.stack(mother_offsets))],1);g=torch.as_tensor(np.stack(gs),device='cuda');y=torch.as_tensor(ys,dtype=torch.float32,device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):loss=F.binary_cross_entropy_with_logits(model(x,g),y)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        previous=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=previous)
        if (step+1)%250==0:row=dict(step=step+1,loss=float(loss.detach()));history.append(row);print('EVENT_TRAIN',row,flush=True)
    delta=float((model.encoder[0].weight.detach()-initial).norm());assert delta>0
    checkpoint=out/'joint_event.pt';torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=cfg),checkpoint);model.eval();outputs=[];del data
    for r in manifest['videos']:
        if r['empty'] or r['split']!='development':continue
        q=p.parent/r['events'];assert sha(q)==r['events_sha256']
        # y intentionally excluded. Thresholds are fitted in the subsequent CPU stage.
        with np.load(q) as d:a={k:d[k] for k in ['mother','delta','geometry']}
        v=p.parent/r['crops'];assert sha(v)==r['crops_sha256'];crops=np.load(v,mmap_mode='r');scores={'temporal':[],'repeated_center':[]}
        with torch.inference_mode():
            for first in range(0,len(a['mother']),cfg['batch']):
                sl=slice(first,first+cfg['batch']);x=torch.as_tensor(np.array(crops[a['mother'][sl]],np.float32),device='cuda');mask=query_masks(a['delta'][sl],'cuda');g=torch.as_tensor(a['geometry'][sl],device='cuda')
                for arm in scores:
                    img=x if arm=='temporal' else x[:,2:3].expand(-1,5,-1,-1,-1)
                    with torch.autocast('cuda',dtype=torch.float16):s=model(torch.cat([img,mask],1),g).sigmoid()
                    scores[arm].append(s.float().cpu().numpy())
        path=out/(r['video']+'_scores.npz');np.savez_compressed(path,**{k:np.concatenate(v) for k,v in scores.items()});outputs.append(dict(video=r['video'],file=path.name,sha256=sha(path)));print('EVENT_SCORES',r['video'],len(a['mother']),flush=True)
    result=dict(status='complete',config=cfg,positive=len(positive),negative=len(negative),optimizer_updates=updates,first_layer_change=delta,history=history,checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),outputs=outputs,seconds=time.monotonic()-start)
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
