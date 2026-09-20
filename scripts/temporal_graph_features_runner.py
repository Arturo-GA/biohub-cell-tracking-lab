import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from biohub_lab.temporal_volume import TemporalVolumeEncoder,gather_temporal

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def one(pattern):
    ps=list(Path('/kaggle/input').rglob(pattern));assert len(ps)==1;return ps[0]

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/temporal_graph_features');root.mkdir(exist_ok=True);torch.set_num_threads(2);assert torch.cuda.is_available()
    config=json.loads((package/'baseline/e033_graph.json').read_text());p=one('temporal_graph_inputs/result.json');data=json.loads(p.read_text());assert data['status']=='complete' and data['config']==config
    trained_path=one('temporal_volume_training/result.json');assert sha(trained_path)==config['training_manifest_sha256'];trained=json.loads(trained_path.read_text());assert trained['status']=='complete'
    outputs=[];models={}
    for f in trained['folds']:
        w=trained_path.parent/f['checkpoint'];assert sha(w)==f['checkpoint_sha256'];saved=torch.load(w,map_location='cpu',weights_only=True)
        net=TemporalVolumeEncoder().cuda().eval();net.load_state_dict(saved['state_dict'],strict=True);models[f['heldout_embryo']]=net
    for item in data['videos']:
        if time.monotonic()-start>600:raise TimeoutError('Feature inference budget exceeded')
        image_path=p.parent/item['image'];graph_path=p.parent/item['graph'];assert sha(image_path)==item['image_sha256'] and sha(graph_path)==item['graph_sha256']
        image=torch.from_numpy(np.load(image_path,allow_pickle=False)).cuda()
        with np.load(graph_path,allow_pickle=False) as d:coords=d['coords']
        centers=torch.as_tensor(np.rint(coords/np.array([1,1,4,4])).astype(np.int64),device='cuda');parts=[];net=models[item['video'].split('_')[0]]
        with torch.inference_mode():
            for first in range(0,len(centers),128):
                crops=gather_temporal(image,centers[first:first+128])
                with torch.autocast('cuda',dtype=torch.float16):h=net(crops.float())
                parts.append(h.float().cpu().numpy())
        path=root/(item['video']+'_features.npy');np.save(path,np.concatenate(parts),allow_pickle=False)
        outputs.append(dict(video=item['video'],file=path.name,sha256=sha(path),graph_sha256=item['graph_sha256']));del image
        print('GRAPH_FEATURES',item['video'],len(coords),flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,outputs=outputs,seconds=time.monotonic()-start,annotations_read=False,training=False),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
