"""CPU: fixed real training crops, with no access to evaluation annotations."""
import json,sys,time
from pathlib import Path
import numpy as np
from ensemble_evaluate_runner import one,sha
from biohub_lab.residual_detector import crop_three,positive_heatmap

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e061_protocol.json').read_text())
    out=Path('/kaggle/working/residual_detector_data');out.mkdir(exist_ok=True)
    p=one('three_lines_data/result.json');assert sha(p)==cfg['real_data_sha256']
    records=[]
    for r in json.loads(p.read_text())['videos']:
        if r['video'] not in cfg['fit']:continue
        assert r['split']=='fit' and r['video'] not in cfg['evaluation']
        q=p.parent/r['truth'];assert sha(q)==r['truth_sha256']
        with np.load(q) as z:coords=z['coords']/[1,1,4,4]
        q=p.parent/r['image'];assert sha(q)==r['image_sha256'];vol=np.load(q,mmap_mode='r')
        rng=np.random.default_rng(cfg['seed']+int(r['video'].split('_')[1],16));n=cfg['crops_per_video']
        indices=rng.choice(len(coords),n,replace=len(coords)<n);xs=[];ys=[]
        for i in indices:
            t=int(coords[i,0]);origin=np.rint(coords[i,1:]+rng.uniform(-8,8,3)).astype(int)-16
            x=crop_three(vol,t,origin);y=positive_heatmap(coords[coords[:,0]==t,1:]-origin)
            assert (y>.3).any();xs.append(x.astype(np.float16));ys.append(y.astype(np.float16))
        path=out/(r['video']+'.npz');np.savez_compressed(path,x=np.array(xs),y=np.array(ys),indices=indices)
        records.append(dict(video=r['video'],file=path.name,sha256=sha(path),samples=n,unique_indices=len(np.unique(indices))))
        print('ENSEMBLE_RESIDUAL_PREP',r['video'],n,flush=True)
    assert sorted(r['video'] for r in records)==sorted(cfg['fit'])
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,evaluation_labels_read=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
