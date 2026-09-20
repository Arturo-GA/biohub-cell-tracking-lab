"""E028: independent transformer segmentation fields, no mask decoding on GPU."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from cellpose_runtime import setup,locate
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/cellpose_fields');root.mkdir(exist_ok=True)
    record=dict(status='starting',experiment='E028',annotations_read=False,optimizer_steps=0)
    def save():(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        assets,prep=setup(package);config=json.loads((package/'baseline/e028_pilot.json').read_text())
        weight=assets/'cpdino-vitb';assert hashlib.sha256(weight.read_bytes()).hexdigest()==prep['weight_sha256']
        import torch
        assert torch.cuda.is_available()
        from cellpose import models
        model=models.CellposeModel(gpu=True,pretrained_model=str(weight),use_bfloat16=False);model.net.eval()
        # Bound neural work between forward batches, not only after a volume.
        def budget(module,args):
            if time.monotonic()-start>config['gpu_deadline_seconds']:raise TimeoutError('Frozen E028 inference budget exceeded')
        hook=model.net.register_forward_pre_hook(budget)
        inputs=locate('nucverse_inputs/result.json');manifest=json.loads(inputs.read_text());lookup={(r['video'],r['frame']):r for r in manifest['frames']};frames=[]
        for selected in config['frames']:
            item=lookup[(selected['video'],selected['frame'])];path=inputs.parent/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            image=np.load(path,allow_pickle=False);tick=time.monotonic()
            with torch.inference_mode():
                _,flows,_=model.eval(image[...,None],channel_axis=3,z_axis=0,do_3D=True,anisotropy=4.,compute_masks=False,batch_size=4,bsize=384,augment=False,normalize=True)
            dP,cellprob=np.asarray(flows[1]),np.asarray(flows[2]);assert dP.shape==(3,*image.shape) and cellprob.shape==image.shape
            assert np.isfinite(dP).all() and np.isfinite(cellprob).all()
            target=root/(path.stem+'_fields.npz');np.savez(target,dP=dP.astype(np.float32),cellprob=cellprob.astype(np.float32))
            frames.append(dict(**item,output=target.name,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=time.monotonic()-tick));print('CELLPOSE_FRAME',frames[-1],flush=True)
        hook.remove();record.update(status='complete',frames=frames,seconds=time.monotonic()-start,weight_sha256=prep['weight_sha256'],config=config,torch=torch.__version__);save();print('CELLPOSE_FIELDS_COMPLETE',json.dumps(record),flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error),seconds=time.monotonic()-start);save();(root/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
