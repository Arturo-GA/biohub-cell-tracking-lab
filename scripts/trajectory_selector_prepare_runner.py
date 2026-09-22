"""CPU coordinate union for selector fit/development; no labels."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from residual_detector_io import one,sha

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e063_protocol.json').read_text());pins=json.loads((package/'baseline/e063_prepare_pins.json').read_text())
    out=Path('/kaggle/working/selector_inputs');out.mkdir(exist_ok=True);p=one('selector_detections/result.json');assert sha(p)==pins['DETECT'];m=json.loads(p.read_text());records=[]
    for r in m['records']:
        q=p.parent/r['file'];assert sha(q)==r['sha256']
        with np.load(q) as z:base=z['primary_coords'];donors=z['adapted_coords'];scores=z['adapted_prob']
        coords=[];mapping=[];prob=[]
        for t in range(r['shape'][0]):
            b=np.flatnonzero(base[:,0]==t);d=np.flatnonzero(donors[:,0]==t)
            if len(b) and len(d):d=d[cKDTree(base[b,1:]*[1.625,.40625,.40625]).query(donors[d,1:]*[1.625,.40625,.40625])[0]>3]
            coords.extend(base[b]);mapping.extend(b);prob.extend(np.ones(len(b)))
            coords.extend(donors[d]);mapping.extend([-1]*len(d));prob.extend(scores[d])
        q=out/r['file'];np.savez_compressed(q,coords=np.asarray(coords,np.int32),mapping=np.asarray(mapping,np.int64),prob=np.asarray(prob,np.float32))
        records.append(dict(video=r['video'],file=q.name,sha256=sha(q),old_nodes=len(base),shape=r['shape']));print('ENSEMBLE_SELECTOR_UNION',r['video'],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,annotations_read=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
