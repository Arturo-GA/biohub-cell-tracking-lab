"""E040 CPU: all genuine training divisions, without the old full-track context filter."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import zarr
from biohub_lab.temporal_data import load_gt
from biohub_lab.expanded_events import event_candidates,geometry
from association_cpu_runner import locate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(package):
    start=time.monotonic();out=Path('/kaggle/working/expanded_event_data');out.mkdir(exist_ok=True);cfg=json.loads((package/'baseline/e040_protocol.json').read_text());inputs=Path('/kaggle/input')
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    source=locate(inputs,'repo/src/biohub_tracking/io.py');sys.path.insert(0,str(source.parents[1]));from biohub_tracking.io import open_dataset
    cached=locate(inputs,'three_lines_data/result.json');cache={r['video']:r for r in json.loads(cached.read_text())['videos']};records=[]
    for split in ['fit','development']:
        for video in cfg[split]:
            coords,edges=load_gt(train/(video+'.geff'));coords=coords/[1,1,4,4];triples,y=event_candidates(coords,edges,training=split=='fit',seed=int(hashlib.sha256(video.encode()).hexdigest()[:8],16))
            if not len(y):records.append(dict(video=video,split=split,positive=0,negative=0,empty=True));continue
            mothers,inverse=np.unique(triples[:,0],return_inverse=True);centers=np.rint(coords[mothers,1:]).astype(int);times=coords[mothers,0].astype(int)
            if video in cache:
                r=cache[video];p=cached.parent/r['image'];assert sha(p)==r['image_sha256'];vol=np.load(p,mmap_mode='r')
            else:
                ip=train/(video+'.zarr');ds=open_dataset(ip,normalize=False,load_image=False,require_tracks=False);lo,hi=float(ds.quantiles['0.001']),float(ds.quantiles['0.999']);raw=zarr.open_group(str(ip),mode='r')['0']
                vol=np.zeros((raw.shape[0],raw.shape[1],len(range(0,raw.shape[2],4)),len(range(0,raw.shape[3],4))),np.float16)
                for t in np.unique(np.clip(times[:,None]+np.arange(-2,3),0,len(vol)-1)):vol[t]=np.clip((np.asarray(raw[int(t),:,::4,::4],np.float32)-lo)/max(hi-lo,1e-6),0,1)
            p=out/(video+'_crops.npy');patches=np.lib.format.open_memmap(p,mode='w+',dtype=np.float16,shape=(len(mothers),5,32,32,32))
            for j,(t,center) in enumerate(zip(times,centers)):
                a=np.pad(np.array(vol[np.clip(np.arange(t-2,t+3),0,len(vol)-1)],np.float32),((0,0),(16,16),(16,16),(16,16)),mode='edge')
                patches[j]=a[(slice(None),*tuple(slice(int(c),int(c)+32) for c in center))]
            patches.flush();del patches,vol
            # Position masks must use the rounded crop origin, not the fractional mother.
            delta=(coords[triples[:,1:],1:]-centers[inverse,None,:]).astype(np.float32)
            geom=geometry(coords[triples[:,1:],1:]-coords[triples[:,0],None,1:]).astype(np.float32)
            q=out/(video+'_events.npz');np.savez_compressed(q,mother=inverse,delta=delta,geometry=geom,y=y,triples=triples,coords=coords,edges=edges)
            rec=dict(video=video,split=split,crops=p.name,crops_sha256=sha(p),events=q.name,events_sha256=sha(q),mothers=len(mothers),positive=int(y.sum()),negative=int((y==0).sum()),empty=False);records.append(rec);print('EVENT_DATA',video,rec['positive'],rec['negative'],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,videos=records,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
