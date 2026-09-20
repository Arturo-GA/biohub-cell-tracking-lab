"""E038 GPU: frozen SpatialDINO localization features and raw-pixel control."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from biohub_lab.spatial_probe import SpatialBackbone,summarize

def one(relative):
    found=[]
    for depth in [1,2,3]:found+=list(Path('/kaggle/input').glob('*/'*depth+relative))
    found=list(dict.fromkeys(found));assert len(found)==1,(relative,found);return found[0]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(package):
    start=time.monotonic();root=Path('/kaggle/working/spatial_probe_features');root.mkdir(exist_ok=True)
    cfg=json.loads((package/'baseline/e038_protocol.json').read_text());p=one('three_lines_data/result.json');manifest=json.loads(p.read_text());assert manifest['config']==cfg
    weight=one('three_lines_audit/backbone.pth');assert sha(weight)==cfg['weight_sha256']
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(380920)
    state=torch.load(weight,weights_only=True,map_location='cpu');model=SpatialBackbone(state,'cuda');records=[]
    for item in manifest['videos']:
        source=p.parent/item['probe'];assert sha(source)==item['probe_sha256']
        with np.load(source) as d:x=d['patches'];offset=d['offset']
        features=[];raw=[]
        with torch.inference_mode():
            for first in range(0,len(x),16):
                a=torch.as_tensor(x[first:first+16].astype(np.float32),device='cuda')
                raw.append(F.adaptive_avg_pool3d(a,8).flatten(1).cpu().numpy())
                a=F.interpolate(a,scale_factor=3,mode='trilinear',align_corners=True)
                with torch.autocast('cuda',dtype=torch.float16):h=summarize(model(a))
                features.append(h.float().cpu().numpy())
        target=root/(item['video']+'.npz');np.savez_compressed(target,spatial=np.concatenate(features),raw=np.concatenate(raw),offset=offset)
        records.append(dict(video=item['video'],split=item['split'],file=target.name,sha256=sha(target)))
        print('SPATIAL_FEATURES',item['video'],len(x),flush=True)
    torch.cuda.synchronize();(root/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,seconds=time.monotonic()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
