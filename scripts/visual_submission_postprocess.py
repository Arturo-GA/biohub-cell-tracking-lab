"""Inference-only exact visual assignment. Reads test images' shapes, never GT."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import tracksdata as td
import zarr
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import COLUMNS,read_and_validate
from biohub_lab.visual_candidates import decode_cache
from biohub_lab.visual_assignment import assign

def postprocess(control,cache,test,output,receipt):
    start=time.monotonic()
    shapes={p.stem:tuple(zarr.open_group(str(p),mode='r')['0'].shape) for p in sorted(test.glob('*.zarr'))}
    if not shapes:raise ValueError('No test images')
    validated=receipt.parent/'harmonic_validated.csv'
    repair=export_control(control/'submission.csv',validated,shapes)
    reference=read_and_validate(validated,shapes);reports={};index=0
    temporary=output.with_suffix('.partial.csv')
    with temporary.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(COLUMNS)
        for name in sorted(shapes):
            paths=list((control/'tracking_repo/predictions').glob('*/unet_transformer/split_0/'+name+'.geff'))
            if len(paths)!=1:raise ValueError('Expected one baseline graph for '+name)
            raw=td.graph.IndexedRXGraph.from_geff(paths[0]);raw=raw[0] if isinstance(raw,tuple) else raw
            raw_nodes={int(r['node_id']):{a:float(r[a]) for a in ('t','z','y','x')} for r in raw.node_attrs().iter_rows(named=True)}
            frames=[]
            for path in sorted((cache/name).glob('*.npz')):
                with np.load(path,allow_pickle=False) as values:frames.append({k:values[k] for k in values.files})
            if not frames and reference[name][1]:raise ValueError('Missing visual evidence for '+name)
            nodes,original=reference[name];visual,audit=decode_cache(raw_nodes,nodes,frames)
            edges,report=assign(nodes,original,visual)
            for k in sorted(nodes):
                v=nodes[k];writer.writerow([index,name,'node',k,*[v[a] for a in ('t','z','y','x')],-1,-1]);index+=1
            for a,b in edges:writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,a,b]);index+=1
            reports[name]=dict(cache=audit,assignment=report,nodes=len(nodes),edges=len(edges))
            print('SUBMISSION_VIDEO_READY',name,len(nodes),len(edges),flush=True)
    # Do not publish submission.csv until every dataset and graph passes validation.
    read_and_validate(temporary,shapes);temporary.replace(output)
    result=dict(status='complete',seconds=time.monotonic()-start,videos=reports,export_repair=repair,
        image_shapes={k:list(v) for k,v in shapes.items()},
        csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),annotations_read=False,
        algorithm='Frozen E022 visual assignment; no temporal penalty or retraining')
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['control','cache','test','output','receipt']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();postprocess(**vars(args))
