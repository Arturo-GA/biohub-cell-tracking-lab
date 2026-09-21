"""CPU union of the frozen base graph and new adapted-detector candidates."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from biohub_lab.submission import read_and_validate
from residual_detector_io import one,sha

def main(package):
    start=time.monotonic();cfg=json.loads((package/'baseline/e062_protocol.json').read_text());out=Path('/kaggle/working/neural_complement_inputs');out.mkdir(exist_ok=True)
    p=one('residual_detector_predictions/result.json');assert sha(p)==cfg['predictions_sha256'];m=json.loads(p.read_text())
    gp=one('residual_detector_evaluation/result.json');assert sha(gp)==cfg['evaluation_sha256'];base=gp.parent/'control.csv'
    assert sha(base)==json.loads((gp.parent/'frozen_predictions.json').read_text())['control']
    shapes={r['video']:r['shape'] for r in m['records']};graphs=read_and_validate(base,shapes);records=[]
    for video,(nodes,edges) in graphs.items():
        r=next(r for r in m['records'] if r['video']==video and r['arm']=='adapted');q=p.parent/r['file'];assert sha(q)==r['sha256']
        with np.load(q) as z:donors=z['coords'];scores=z['prob']
        ids=np.array(sorted(nodes));original=np.array([[nodes[int(k)][a] for a in ('t','z','y','x')] for k in ids]);coords=[];mapping=[];prob=[]
        for t in range(shapes[video][0]):
            b=np.flatnonzero(original[:,0]==t);d=np.flatnonzero(donors[:,0]==t)
            if len(b) and len(d):d=d[cKDTree(original[b,1:]*[1.625,.40625,.40625]).query(donors[d,1:]*[1.625,.40625,.40625])[0]>3]
            coords.extend(original[b]);mapping.extend(ids[b]);prob.extend(np.ones(len(b)))
            coords.extend(donors[d]);mapping.extend([-1]*len(d));prob.extend(scores[d])
        q=out/(video+'.npz');np.savez_compressed(q,coords=np.asarray(coords,np.int32),mapping=np.asarray(mapping,np.int64),prob=np.asarray(prob,np.float32))
        records.append(dict(video=video,file=q.name,sha256=sha(q),old_nodes=len(nodes),new_candidates=len(coords)-len(nodes),shape=shapes[video]));print('ENSEMBLE_NEURAL_UNION',records[-1],flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,seconds=time.monotonic()-start,annotations_read=False),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
