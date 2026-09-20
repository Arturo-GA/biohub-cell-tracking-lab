"""E030 GPU only: frozen actual DINO patch descriptors, no optimizer."""
import hashlib, json, time, traceback
from pathlib import Path
import numpy as np
from cellpose_runtime import setup, locate


def main(package):
    start = time.monotonic(); root = Path('/kaggle/working/image_identity_features'); root.mkdir(exist_ok=True)
    record = dict(status='starting',optimizer_steps=0,styles_used=False)
    def save(): (root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        import torch
        from biohub_lab.image_identity import extract_tokens
        assets, prep = setup(package)
        from cellpose import models
        assert torch.cuda.is_available()
        torch.set_num_threads(2)
        weight = assets/'cpdino-vitb'; assert hashlib.sha256(weight.read_bytes()).hexdigest() == prep['weight_sha256']
        model = models.CellposeModel(gpu=True,pretrained_model=str(weight),use_bfloat16=False)
        state = torch.load(weight,map_location='cpu',weights_only=True); defaults = model.net.state_dict()
        missing = set(defaults)-set(state)
        assert missing == {'diam_labels','diam_mean'} and not set(state)-set(defaults)
        for key in missing:
            assert not dict(model.net.named_parameters())[key].requires_grad and torch.all(defaults[key]==30)
            state[key] = defaults[key]
        model.net.load_state_dict(state,strict=True); model.net.eval(); model.net.requires_grad_(False)
        config = json.loads((package/'baseline/e030_image_identity.json').read_text())
        source = locate('image_identity_data/result.json'); data = json.loads(source.read_text())
        assert data['status'] == 'complete' and data['config'] == config
        manifest = []; deterministic_error = None
        for item in data['videos']:
            path = source.parent/item['file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            with np.load(path,allow_pickle=False) as archive: crops = archive['crops']
            views = crops.reshape(-1,1,64,64); output = []
            for first in range(0,len(views),config['feature_batch_size']):
                if time.monotonic()-start > config['gpu_deadline_seconds']: raise TimeoutError('E030 frozen GPU budget exceeded')
                batch = torch.as_tensor(views[first:first+config['feature_batch_size']].astype(np.float32),device='cuda')
                descriptor = extract_tokens(model.net,batch)
                if deterministic_error is None:
                    repeated = extract_tokens(model.net,batch)
                    deterministic_error = float((descriptor-repeated).abs().max())
                    assert deterministic_error < 1e-6, 'Random descriptors detected'
                output.append(descriptor.cpu().numpy().astype(np.float16))
            features = np.concatenate(output).reshape(len(crops),-1)
            assert features.shape == (len(crops),config['features']) and np.isfinite(features).all()
            target = root/(item['video']+'.npy'); np.save(target,features,allow_pickle=False)
            manifest.append(dict(video=item['video'],split=item['split'],file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_sha256=item['sha256'],shape=list(features.shape)))
            print('DINO_TOKENS',item['video'],features.shape,time.monotonic()-start,flush=True)
        record.update(status='complete',videos=manifest,seconds=time.monotonic()-start,deterministic_max_error=deterministic_error,weight_sha256=prep['weight_sha256'],input_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),config=config,torch=torch.__version__)
        save();print('DINO_TOKENS_COMPLETE',record['seconds'],flush=True)
    except Exception as error:
        record.update(status='failed',seconds=time.monotonic()-start,error=str(error));save();(root/'error.txt').write_text(traceback.format_exc());raise

if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))
