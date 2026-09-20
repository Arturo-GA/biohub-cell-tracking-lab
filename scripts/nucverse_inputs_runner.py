"""E024 CPU: freeze eight full frames and reference coordinates for a new detector."""
import csv,json,time,hashlib
from pathlib import Path
import numpy as np
import zarr,tracksdata as td
from association_cpu_runner import locate,evaluation_dir
from biohub_lab.evaluate import shapes_for
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import read_and_validate

def main(package):
    root=Path('/kaggle/working/nucverse_inputs');root.mkdir(exist_ok=True);start=time.monotonic()
    config=json.loads((package/'baseline/e024_benchmark.json').read_text())
    inputs=Path('/kaggle/input')
    train=next(p/'train' for p in [inputs/'biohub-cell-tracking-during-development',inputs/'competitions/biohub-cell-tracking-during-development'] if (p/'train').exists())
    data=evaluation_dir(train,config['videos'],root/'data');shapes=shapes_for(data)
    control=locate(inputs,'calibration_control/result.json').parent/'harmonic_control/submission.csv'
    with control.open(newline='') as inp,(root/'subset.csv').open('w',newline='') as out:
        reader=csv.DictReader(inp);writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader();index=0
        for row in reader:
            if row['dataset'] in shapes:
                row['id']=index;writer.writerow(row);index+=1
    export_control(root/'subset.csv',root/'control.csv',shapes)
    reference=read_and_validate(root/'control.csv',shapes)
    manifest=[]
    for name in config['videos']:
        volume=zarr.open_group(str(train/(name+'.zarr')),mode='r')['0']
        graph=td.graph.IndexedRXGraph.from_geff(train/(name+'.geff'));graph=graph[0] if isinstance(graph,tuple) else graph
        gt=list(graph.node_attrs().iter_rows(named=True));nodes=reference[name][0]
        for frame in config['frames']:
            image=np.asarray(volume[frame]);path=root/(name+'_'+str(frame)+'.npy');np.save(path,image,allow_pickle=False)
            rows={key:np.array([[r[a] for a in ('z','y','x')] for r in values if int(r['t'])==frame],float).reshape(-1,3) for key,values in [('truth',gt),('harmonic',nodes.values())]}
            np.savez_compressed(root/(path.stem+'_reference.npz'),**rows)
            manifest.append(dict(video=name,frame=frame,file=path.name,shape=list(image.shape),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record=dict(status='complete',frames=manifest,selection=config,seconds=time.monotonic()-start,
        annotations_used_to_select_frames=False,scope='Development detector benchmark, not an official tracking or leaderboard score')
    (root/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
