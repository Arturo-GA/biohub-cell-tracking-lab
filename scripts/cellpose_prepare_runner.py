"""E028 CPU-only public assets and strict model compatibility check."""
import os,sys,subprocess,json,hashlib,time,traceback,urllib.request
from pathlib import Path
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/cellpose_assets');root.mkdir(exist_ok=True)
    record=dict(status='starting',gpu=False,training=False,competition_data_read=False)
    def save():(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        config=json.loads((package/'baseline/e028_cellpose.json').read_text());wheels=root/'wheels';wheels.mkdir(exist_ok=True)
        requirements=['cellpose=='+config['cellpose_version'],'natsort','fastremap','fill-voids','roifile','segment_anything','imagecodecs']
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:','--dest',str(wheels),*requirements],check=True)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),'https://github.com/facebookresearch/dinov3/archive/'+config['dinov3_commit']+'.zip'],check=True)
        subprocess.run([sys.executable,'-m','pip','install','--no-deps',*[str(p) for p in wheels.glob('*.whl')]],check=True)
        weight=root/'cpdino-vitb';url='https://huggingface.co/mouseland/cellpose-sam/resolve/'+config['hf_revision']+'/cpdino-vitb'
        with urllib.request.urlopen(url,timeout=60) as source,weight.open('wb') as target:
            while chunk:=source.read(1024*1024):target.write(chunk)
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
        import torch,numpy as np
        assert not torch.cuda.is_available()
        from cellpose import models
        model=models.CellposeModel(gpu=False,pretrained_model=str(weight),use_bfloat16=False)
        state=torch.load(weight,map_location='cpu',weights_only=True)
        defaults=model.net.state_dict();missing=set(defaults)-set(state)
        assert missing=={'diam_labels','diam_mean'} and not set(state)-set(defaults)
        for key in sorted(missing):
            assert not dict(model.net.named_parameters())[key].requires_grad and torch.all(defaults[key]==30)
            state[key]=defaults[key]
        model.net.load_state_dict(state,strict=True);model.net.eval()
        with torch.inference_mode():output=model.net(torch.zeros((1,3,64,64),dtype=torch.float32))
        field=output[0] if isinstance(output,(tuple,list)) else output
        assert tuple(field.shape)==(1,3,64,64) and torch.isfinite(field).all()
        files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(wheels.glob('*.whl'))]
        record.update(status='complete',config=config,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),weight_bytes=weight.stat().st_size,wheels=files,strict_load=True,metadata_defaults_added=sorted(missing),output_shape=list(field.shape),parameters=sum(p.numel() for p in model.net.parameters()),torch=torch.__version__,numpy=np.__version__,seconds=time.monotonic()-start,scope='Public assets and compatibility only; no Biohub accuracy measured')
        save();print('CELLPOSE_PREPARED',json.dumps(record),flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),seconds=time.monotonic()-start);save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main(Path(sys.argv[1]))
