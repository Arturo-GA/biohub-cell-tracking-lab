"""E031 CPU preparation: all frames of the fixed calibration cohort."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import zarr,tracksdata as td

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/dense_detector_inputs');root.mkdir(exist_ok=True)
    config=json.loads((package/'baseline/e031_dense_detector.json').read_text())
    inputs=Path('/kaggle/input')
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    repos=list(inputs.rglob('repo/src/biohub_tracking/io.py'));assert len(repos)==1
    sys.path.insert(0,str(repos[0].parents[1]))
    from biohub_tracking.io import open_dataset
    records=[];keys=td.DEFAULT_ATTR_KEYS
    for video in config['videos']:
        ds=open_dataset(train/(video+'.zarr'),normalize=False,load_image=False,require_tracks=False)
        low,high=float(ds.quantiles['0.001']),float(ds.quantiles['0.999'])
        image=zarr.open_group(str(train/(video+'.zarr')),mode='r')['0']
        path=root/(video+'.npy')
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(image.shape[0],image.shape[1],len(range(0,image.shape[2],4)),len(range(0,image.shape[3],4))))
        for t in range(image.shape[0]):arr[t]=np.maximum((np.asarray(image[t,::1,::4,::4],np.float32)-low)/(high-low+1e-6),0)
        arr.flush();del arr
        graph=td.graph.IndexedRXGraph.from_geff(train/(video+'.geff'));graph=graph[0] if isinstance(graph,tuple) else graph
        nodes=list(graph.node_attrs().iter_rows(named=True));edges=list(graph.edge_attrs(attr_keys=[keys.EDGE_SOURCE,keys.EDGE_TARGET]).iter_rows(named=True))
        truth={'nodes':[[int(r[keys.NODE_ID]),int(r['t']),float(r['z']),float(r['y']),float(r['x'])] for r in nodes],'edges':[[int(r[keys.EDGE_SOURCE]),int(r[keys.EDGE_TARGET])] for r in edges]}
        (root/(video+'_truth.json')).write_text(json.dumps(truth))
        records.append(dict(video=video,file=path.name,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),shape=list(image.shape),quantiles=[low,high]))
        print('DENSE_PREP',video,flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,videos=records,seconds=time.monotonic()-start,selection_uses_errors=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
