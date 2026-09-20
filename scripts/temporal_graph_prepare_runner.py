"""CPU image normalization and Harmonic node order for full-graph integration."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from association_cpu_runner import locate
from biohub_lab.submission import read_and_validate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(package):
    start=time.monotonic();root=Path('/kaggle/working/temporal_graph_inputs');root.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    config=json.loads((package/'baseline/e033_graph.json').read_text())
    manifest_path=locate(inputs,'dense_detector_inputs/result.json');assert sha(manifest_path)==config['image_manifest_sha256']
    images=json.loads(manifest_path.read_text());control=locate(inputs,'calibration_compare/harmonic_validated.csv');assert sha(control)==config['control_sha256']
    shapes={r['video']:tuple(r['shape']) for r in images['videos']};graphs=read_and_validate(control,shapes);records=[]
    for item in images['videos']:
        nodes,edges=graphs[item['video']];ids=np.array(sorted(nodes),np.int64);coords=np.array([[nodes[int(n)][a] for a in ('t','z','y','x')] for n in ids],np.float32)
        source=manifest_path.parent/item['file'];assert sha(source)==item['sha256'];raw=np.load(source,allow_pickle=False)
        padded=np.pad(np.log1p(raw.astype(np.float32)),((0,0),(8,8),(8,8),(8,8)),mode='edge').astype(np.float16)
        path=root/(item['video']+'_padded.npy');np.save(path,padded,allow_pickle=False);del raw,padded
        graph=root/(item['video']+'_graph.npz');np.savez_compressed(graph,ids=ids,coords=coords,edges=np.asarray(edges,np.int64).reshape(-1,2))
        r=dict(video=item['video'],image=path.name,image_sha256=sha(path),graph=graph.name,graph_sha256=sha(graph),nodes=len(ids));records.append(r);print('GRAPH_PREP',item['video'],len(ids),flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='complete',config=config,videos=records,seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
