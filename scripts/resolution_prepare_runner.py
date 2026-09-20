"""E042 CPU native-XY patches using exactly E038 centers and offsets."""
import json,sys,time
from pathlib import Path
import numpy as np
import zarr
from spatial_probe_features_runner import one,sha
from association_cpu_runner import locate

def main(package):
    start=time.monotonic();out=Path('/kaggle/working/resolution_data');out.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    p=one('three_lines_data/result.json');m=json.loads(p.read_text());assert sha(p)=='b8df897a2b8496d1b1a37018147df03f9db25b6c55806328e68b675b6ea3454a'
    train=next(q/'train' for q in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (q/'train').exists())
    source=locate(inputs,'repo/src/biohub_tracking/io.py');sys.path.insert(0,str(source.parents[1]));from biohub_tracking.io import open_dataset
    records=[]
    for r in m['videos']:
        with np.load(p.parent/r['probe']) as d:indices=d['indices'];offset=d['offset'];low=d['patches']
        with np.load(p.parent/r['truth']) as d:all_coords=d['coords'];edges=d['edges']
        parents={int(b):int(a) for a,b in edges};counts=np.bincount(edges[:,0].astype(int),minlength=len(all_coords))
        keep=np.array([int(n) in parents and counts[parents[int(n)]]==1 for n in indices])
        indices=indices[keep];offset=offset[keep];low=low[keep];coords=all_coords[indices]
        if not len(indices):print('RESOLUTION_SKIP',r['video'],'no known-parent queries',flush=True);continue
        query=all_coords[[parents[int(n)] for n in indices]]
        guess=np.rint(coords[:,1:]/[1,4,4]-offset).astype(int)
        ip=train/(r['video']+'.zarr');ds=open_dataset(ip,normalize=False,load_image=False,require_tracks=False);lo,hi=float(ds.quantiles['0.001']),float(ds.quantiles['0.999']);raw=zarr.open_group(str(ip),mode='r')['0']
        high=np.empty((len(indices),2,16,64,64),np.float16)
        times=np.stack([coords[:,0],query[:,0]],1).astype(int)
        centers=np.stack([guess,np.rint(query[:,1:]/[1,4,4]).astype(int)],1)*[1,4,4]
        for t in np.unique(times):
            frame=np.clip((np.asarray(raw[t],np.float32)-lo)/max(hi-lo,1e-6),0,1)
            for j,channel in np.argwhere(times==t):
                c=centers[j,channel];limits=np.asarray(frame.shape)-[1,4,4]
                ix=[np.clip(np.arange(n)+int(v)-n//2,0,limit) for n,v,limit in zip([16,64,64],c,limits)]
                high[j,channel]=frame[np.ix_(*ix)]
        assert np.array_equal(high[:,0:1,:,::4,::4],low),r['video']
        path=out/(r['video']+'.npz');np.savez_compressed(path,high=high,offset=offset)
        records.append(dict(video=r['video'],split=r['split'],file=path.name,samples=len(indices),original_samples=len(keep),known_parent_query=True,sha256=sha(path)));print('RESOLUTION_PREP',r['video'],len(indices),flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',videos=records,seconds=time.monotonic()-start,coarse_samples_exactly_equal=True),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
