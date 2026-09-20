"""Inventory genuine annotated events and public SpatialDINO assets on CPU."""
import hashlib,json,sys,time,urllib.request
from pathlib import Path
import numpy as np
from biohub_lab.temporal_data import load_gt

def main(package):
    start=time.monotonic(); out=Path('/kaggle/working/three_lines_audit');out.mkdir(exist_ok=True)
    inputs=Path('/kaggle/input');train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    old=json.loads((package/'baseline/e035_division_sequence.json').read_text()); seen=set(sum([old[k] for k in ['fit','development','validation']],[])); rows=[]
    for p in sorted(train.glob('*.geff')):
        coords,edges=load_gt(p);counts=np.bincount(edges[:,0],minlength=len(coords)) if len(edges) else np.zeros(len(coords),int)
        rows.append(dict(video=p.stem,nodes=len(coords),edges=len(edges),divisions=int((counts==2).sum()),previous_cohort=p.stem in seen))
    url='https://spatialdino.s3.amazonaws.com/models/spatial_dino/step=249999/backbone.pth'
    asset=dict(url=url,status='unavailable')
    try:
        urllib.request.urlretrieve(url,out/'backbone.pth')
        asset.update(status='downloaded',sha256=hashlib.sha256((out/'backbone.pth').read_bytes()).hexdigest(),bytes=(out/'backbone.pth').stat().st_size)
    except Exception as e:asset['error']=str(e)
    result=dict(status='complete',videos=rows,spatialdino=asset,seconds=time.monotonic()-start,annotation_scope='Consecutive annotated edges only; unannotated cells are unknown, never negatives.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('AUDIT_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
