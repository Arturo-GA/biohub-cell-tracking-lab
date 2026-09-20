"""E033 GPU: train the3D encoder itself, then export heldout embeddings."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.temporal_volume import TemporalVolumeEncoder,logits

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/temporal_volume_training');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e033_temporal_volume.json').read_text())
    pins=json.loads((package/'baseline/e033_input_pin.json').read_text())
    paths=list(Path('/kaggle/input').rglob('temporal_volume_data/result.json'));assert len(paths)==1;source=paths[0]
    assert sha(source)==pins['manifest_sha256'];manifest=json.loads(source.read_text());assert manifest['status']=='complete' and manifest['config']==config
    torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cudnn.benchmark=True
    result=dict(status='starting',config=config,folds=[],input_manifest_sha256=sha(source),encoder_trained=True,leaderboard_submitted=False)
    def save():(root/'result.json').write_text(json.dumps(result,indent=2))
    def checktime():
        if time.monotonic()-start>config['deadline_seconds']:raise TimeoutError('E033 GPU process budget reached')
    def load(item,training):
        p=source.parent/item['crops'];q=source.parent/item['candidates']
        assert sha(p)==item['crops_sha256'] and sha(q)==item['candidates_sha256']
        with np.load(q,allow_pickle=False) as arrays:
            d={k:arrays[k] for k in (['indices','distance','labels','mask'] if training else ['indices','distance'])}
        d['crops']=np.load(p,mmap_mode='r',allow_pickle=False)
        if training:d['eligible']=np.flatnonzero(d['mask'].sum(1)>1)
        return d
    def embed(model,crops):
        parts=[];model.eval()
        with torch.inference_mode():
            for first in range(0,len(crops),128):
                checktime();x=torch.from_numpy(np.array(crops[first:first+128],np.float32)).cuda()
                with torch.autocast('cuda',dtype=torch.float16):h=model(x)
                parts.append(h.float().cpu().numpy())
        return np.concatenate(parts)
    save()
    try:
        for embryo in ['44b6','6bba']:
            checktime();torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);rng=np.random.default_rng(config['seed'])
            training=[r for r in manifest['videos'] if not r['video'].startswith(embryo)]
            testing=[r for r in manifest['videos'] if r['video'].startswith(embryo)]
            assert len(training)==len(testing)==8
            model=TemporalVolumeEncoder().cuda();initial=model.net[0].weight.detach().clone()
            random={}
            for r in testing:
                data=load(r,False);random[r['video']]=embed(model,data['crops']);del data
            fit=[load(r,True) for r in training];fit=[d for d in fit if len(d['eligible'])];assert fit
            opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
            model.train();history=[];updates=0
            for step in range(1,config['steps']+1):
                checktime();d=fit[int(rng.integers(len(fit)))];ix=rng.choice(d['eligible'],config['batch'],replace=True)
                nodes=d['indices'][ix];unique,inverse=np.unique(nodes,return_inverse=True)
                x=torch.from_numpy(np.array(d['crops'][unique],np.float32)).cuda()
                for dim in [2,3,4]:
                    if rng.random()<.5:x=torch.flip(x,[dim])
                x=x*(.8+.4*torch.rand((len(x),1,1,1,1),device='cuda'))
                distance=torch.as_tensor(d['distance'][ix],device='cuda');target=torch.as_tensor(d['labels'][ix],device='cuda');mask=torch.as_tensor(d['mask'][ix],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    h=model(x)[torch.as_tensor(inverse,device='cuda')].reshape(config['batch'],13,64)
                    combined,visual=logits(h,distance,config['temperature'])
                    loss=F.cross_entropy(combined.masked_fill(~mask,-1e4),target)+.5*F.cross_entropy(visual.masked_fill(~mask,-1e4),target)
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old_scale=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=old_scale)
                if step%200==0:
                    hrow=dict(step=step,loss=float(loss.detach()));history.append(hrow)
                    result.update(status='training',heldout_embryo=embryo,step=step,seconds=time.monotonic()-start);save();print('VOLUME_TRAIN',embryo,hrow,flush=True)
            delta=float((model.net[0].weight.detach()-initial).norm());assert delta>0
            checkpoint=root/(embryo+'_encoder.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,heldout_embryo=embryo,steps=step),checkpoint)
            del fit
            outputs=[]
            for r in testing:
                data=load(r,False);trained=embed(model,data['crops']);assert np.isfinite(trained).all()
                path=root/(r['video']+'_embeddings.npz');np.savez_compressed(path,trained=trained,random=random[r['video']])
                outputs.append(dict(video=r['video'],file=path.name,sha256=sha(path)));del data
            result['folds'].append(dict(heldout_embryo=embryo,training_videos=[r['video'] for r in training],steps=step,optimizer_updates=updates,first_conv_change_l2=delta,history=history,checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),outputs=outputs))
            save();print('VOLUME_FOLD_COMPLETE',embryo,delta,flush=True)
        result.update(status='complete',seconds=time.monotonic()-start);save();print('VOLUME_TRAIN_COMPLETE',flush=True)
    except Exception as e:
        result.update(status='failed',error=str(e),seconds=time.monotonic()-start);save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
