"""E031 neural inference only, pinned public detector, two BN semantics."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch

def one(root,pattern):
    values=list(root.rglob(pattern));assert len(values)==1,(pattern,len(values));return values[0]

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/dense_detector_predictions');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e031_dense_detector.json').read_text())
    inputs=Path('/kaggle/input');manifest_path=one(inputs,'dense_detector_inputs/result.json')
    manifest=json.loads(manifest_path.read_text());assert manifest['status']=='complete' and manifest['config']==config
    source=one(inputs,'model_v5.py');pin=json.loads((package/'baseline/e031_source_pin.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==pin['model_sha256']
    sys.path.insert(0,str(source.parent));from model_v5 import StrongUNet3D3Level
    weight=source.parent/'00000030.pth';weight_sha=hashlib.file_digest(weight.open('rb'),'sha256').hexdigest()
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    assert torch.cuda.is_available();torch.set_num_threads(2)
    model=StrongUNet3D3Level(in_channels=1,channels=(64,128,256),node_channels=1,gradient_checkpointing=False).cuda()
    state=checkpoint['model_state_dict'];model.load_state_dict(state,strict=True)
    results=[]
    for mode in config['modes']:
        model.load_state_dict(state,strict=True);model.train(mode=='batch_bn')
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.momentum=0.
        for item in manifest['videos']:
            if time.monotonic()-start>config['gpu_deadline_seconds']:raise TimeoutError('Neural inference time budget exceeded')
            path=manifest_path.parent/item['file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==item['sha256']
            image=np.load(path,mmap_mode='r',allow_pickle=False);coordinates=[];features=[];probabilities=[];tick=time.monotonic()
            for t in range(0,len(image),config['batch_size']):
                x=torch.from_numpy(np.array(image[t:t+config['batch_size']],dtype=np.float32)).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    pyramid,logits=model(x);p=torch.sigmoid(logits.float())
                    pooled=torch.nn.functional.max_pool3d(p[:,None],3,1,1)[:,0]
                    for b in range(len(x)):
                        idx=((p[b]>=pooled[b])&(p[b]>=config['threshold'])).nonzero()
                        z,y,xx=idx.unbind(1)
                        f=pyramid[0][b,:,z,y,xx].T
                        c=idx.float()*torch.tensor([1,4,4],device='cuda')
                        coordinates.append(torch.cat([torch.full((len(c),1),t+b,device='cuda'),c],1).cpu().numpy())
                        features.append(f.cpu().numpy().astype(np.float16));probabilities.append(p[b,z,y,xx].cpu().numpy())
                del pyramid,logits,p,pooled,x
            target=root/(item['video']+'_'+mode+'.npz')
            np.savez_compressed(target,coords=np.concatenate(coordinates),features=np.concatenate(features),probability=np.concatenate(probabilities))
            report=dict(video=item['video'],mode=mode,file=target.name,nodes=sum(map(len,coordinates)),seconds=time.monotonic()-tick,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest())
            results.append(report);print('DENSE_GPU',json.dumps(report),flush=True)
    result=dict(status='complete',config=config,files=results,seconds=time.monotonic()-start,weight_sha256=weight_sha,source_sha256=pin['model_sha256'],annotations_read=False,optimizer_steps=0,submission=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print('DENSE_GPU_COMPLETE',flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
